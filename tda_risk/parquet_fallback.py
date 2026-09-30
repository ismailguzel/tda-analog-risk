from pathlib import Path
from thrift.transport.TTransport import TMemoryBuffer
from thrift.protocol.TCompactProtocol import TCompactProtocol
from thrift.Thrift import TType
import struct, ctypes, ctypes.util, math, datetime

# parquet enums
PT_BOOLEAN=0; PT_INT32=1; PT_INT64=2; PT_INT96=3; PT_FLOAT=4; PT_DOUBLE=5; PT_BYTE_ARRAY=6; PT_FIXED_LEN_BYTE_ARRAY=7
ENC_PLAIN=0; ENC_PLAIN_DICTIONARY=2; ENC_RLE=3; ENC_BIT_PACKED=4; ENC_RLE_DICTIONARY=8
PAGE_DATA=0; PAGE_INDEX=1; PAGE_DICTIONARY=2; PAGE_DATA_V2=3
CODEC_UNCOMPRESSED=0; CODEC_SNAPPY=1

class Snappy:
    def __init__(self):
        name=ctypes.util.find_library('snappy') or 'libsnappy.so.1'
        self.lib=ctypes.CDLL(name)
        self.lib.snappy_uncompressed_length.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_size_t)]
        self.lib.snappy_uncompressed_length.restype=ctypes.c_int
        self.lib.snappy_uncompress.argtypes=[ctypes.c_char_p,ctypes.c_size_t,ctypes.c_char_p,ctypes.POINTER(ctypes.c_size_t)]
        self.lib.snappy_uncompress.restype=ctypes.c_int
    def decompress(self,b):
        n=ctypes.c_size_t()
        rc=self.lib.snappy_uncompressed_length(b,len(b),ctypes.byref(n)); assert rc==0,rc
        out=ctypes.create_string_buffer(n.value); m=ctypes.c_size_t(n.value)
        rc=self.lib.snappy_uncompress(b,len(b),out,ctypes.byref(m)); assert rc==0,rc
        return out.raw[:m.value]
SNAPPY=Snappy()

def read_value(p,ttype):
    if ttype==TType.BOOL:return p.readBool()
    if ttype==TType.BYTE:return p.readByte()
    if ttype==TType.I16:return p.readI16()
    if ttype==TType.I32:return p.readI32()
    if ttype==TType.I64:return p.readI64()
    if ttype==TType.DOUBLE:return p.readDouble()
    if ttype==TType.STRING:return p.readBinary()
    if ttype==TType.LIST:
        et,n=p.readListBegin(); v=[read_value(p,et) for _ in range(n)]; p.readListEnd(); return v
    if ttype==TType.SET:
        et,n=p.readSetBegin(); v=[read_value(p,et) for _ in range(n)]; p.readSetEnd(); return v
    if ttype==TType.MAP:
        kt,vt,n=p.readMapBegin(); v=[(read_value(p,kt),read_value(p,vt)) for _ in range(n)]; p.readMapEnd(); return v
    if ttype==TType.STRUCT:return read_struct(p)
    raise ValueError(ttype)
def read_struct(p):
    p.readStructBegin(); d={}
    while True:
        _,tt,fid=p.readFieldBegin()
        if tt==TType.STOP:break
        d[fid]=read_value(p,tt); p.readFieldEnd()
    p.readStructEnd(); return d

def parse_struct_at(b,off):
    t=TMemoryBuffer(b[off:]); p=TCompactProtocol(t); d=read_struct(p); return d,t.cstringio_buf.tell()

def read_uvarint(buf,pos):
    val=0; shift=0
    while True:
        x=buf[pos]; pos+=1; val |= (x & 0x7f)<<shift
        if x<128:return val,pos
        shift+=7

def decode_rle_bitpacked(buf, bit_width, count=None):
    vals=[]; pos=0; byte_width=(bit_width+7)//8
    while pos<len(buf) and (count is None or len(vals)<count):
        header,pos=read_uvarint(buf,pos)
        if header & 1 == 0:
            run=header>>1
            if byte_width:
                v=int.from_bytes(buf[pos:pos+byte_width],'little'); pos+=byte_width
            else:v=0
            vals.extend([v]*run)
        else:
            groups=header>>1; n=groups*8; nbits=n*bit_width; nbytes=(nbits+7)//8
            data=buf[pos:pos+nbytes]; pos+=nbytes
            if bit_width==0: vals.extend([0]*n)
            else:
                acc=int.from_bytes(data,'little')
                mask=(1<<bit_width)-1
                vals.extend((acc>>(i*bit_width))&mask for i in range(n))
    if count is not None:return vals[:count],pos
    return vals,pos

def parse_plain(buf,ptype,count,type_length=None):
    vals=[]; pos=0
    if ptype==PT_BYTE_ARRAY:
        for _ in range(count):
            n=struct.unpack_from('<I',buf,pos)[0]; pos+=4; vals.append(buf[pos:pos+n]); pos+=n
    elif ptype==PT_INT32:
        vals=list(struct.unpack_from('<'+'i'*count,buf,pos)); pos+=4*count
    elif ptype==PT_INT64:
        vals=list(struct.unpack_from('<'+'q'*count,buf,pos)); pos+=8*count
    elif ptype==PT_FLOAT:
        vals=list(struct.unpack_from('<'+'f'*count,buf,pos)); pos+=4*count
    elif ptype==PT_DOUBLE:
        vals=list(struct.unpack_from('<'+'d'*count,buf,pos)); pos+=8*count
    elif ptype==PT_BOOLEAN:
        for i in range(count): vals.append(bool((buf[i//8]>>(i%8))&1))
        pos=(count+7)//8
    elif ptype==PT_FIXED_LEN_BYTE_ARRAY:
        assert type_length
        for _ in range(count): vals.append(buf[pos:pos+type_length]); pos+=type_length
    else: raise NotImplementedError(ptype)
    return vals,pos

class MiniParquet:
    def __init__(self,path):
        self.path=Path(path); self.b=self.path.read_bytes(); assert self.b[:4]==b'PAR1' and self.b[-4:]==b'PAR1'
        mlen=struct.unpack('<I',self.b[-8:-4])[0]; self.meta,_=parse_struct_at(self.b,len(self.b)-8-mlen)
        self.schema=self.meta[2]
        self.names=[x[4].decode() if isinstance(x[4],bytes) else x[4] for x in self.schema[1:]]
        self.schema_map={(x[4].decode() if isinstance(x[4],bytes) else x[4]):x for x in self.schema[1:]}
        self.row_groups=self.meta[4]
    def columns(self):return self.names
    def _decompress(self,payload,codec,expected=None):
        if codec==CODEC_UNCOMPRESSED:return payload
        if codec==CODEC_SNAPPY:return SNAPPY.decompress(payload)
        raise NotImplementedError(('codec',codec))
    def read_column(self,name):
        out=[]
        for rg in self.row_groups:
            cc=None
            for c in rg[1]:
                md=c[3]
                path=[x.decode() if isinstance(x,bytes) else x for x in md[3]]
                if path==[name]:cc=c;break
            if cc is None:raise KeyError(name)
            out.extend(self._read_chunk(cc, self.schema_map[name]))
        return out
    def _read_chunk(self,cc,se):
        md=cc[3]; ptype=md[1]; codec=md[4]; nvalues=md[5]
        start=md.get(11,md[9]); end=start+md[7]
        pos=start; dictionary=None; values=[]
        max_def=1 if se.get(3)==1 else 0 # optional
        type_length=se.get(2)
        while pos<end and len(values)<nvalues:
            hdr,hlen=parse_struct_at(self.b,pos); pos+=hlen
            csize=hdr[3]; payload=self.b[pos:pos+csize]; pos+=csize
            ptype_page=hdr[1]
            if ptype_page==PAGE_DICTIONARY:
                data=self._decompress(payload,codec,hdr[2]); dh=hdr[7]; cnt=dh[1]; enc=dh[2]
                assert enc in (ENC_PLAIN,ENC_PLAIN_DICTIONARY)
                dictionary,_=parse_plain(data,ptype,cnt,type_length)
            elif ptype_page==PAGE_DATA:
                data=self._decompress(payload,codec,hdr[2]); h=hdr[5]; num=h[1]; enc=h[2]
                cur=0
                # max rep is zero => no repetition-level bytes
                if max_def>0:
                    l=struct.unpack_from('<I',data,cur)[0]; cur+=4
                    defs,_=decode_rle_bitpacked(data[cur:cur+l],1,num); cur+=l
                else: defs=[0]*num
                nn=sum(1 for x in defs if x==max_def) if max_def>0 else num
                valbuf=data[cur:]
                if enc in (ENC_RLE_DICTIONARY,ENC_PLAIN_DICTIONARY):
                    bw=valbuf[0]; inds,_=decode_rle_bitpacked(valbuf[1:],bw,nn); raw=[dictionary[i] for i in inds]
                elif enc==ENC_PLAIN:
                    raw,_=parse_plain(valbuf,ptype,nn,type_length)
                else: raise NotImplementedError(('encoding',enc))
                it=iter(raw)
                values.extend(next(it) if (max_def==0 or d==max_def) else None for d in defs)
            elif ptype_page==PAGE_DATA_V2:
                h=hdr[8]; num=h[1]; nulls=h[2]; rep_len=h[6]; def_len=h[5]; enc=h[4]; iscomp=h.get(7,True)
                # levels are uncompressed; values may be compressed
                rep=payload[:rep_len]; deff=payload[rep_len:rep_len+def_len]; valcomp=payload[rep_len+def_len:]
                defs,_=decode_rle_bitpacked(deff,1,num) if max_def>0 else ([0]*num,0)
                valbuf=self._decompress(valcomp,codec) if iscomp else valcomp
                nn=num-nulls
                if enc in (ENC_RLE_DICTIONARY,ENC_PLAIN_DICTIONARY):
                    bw=valbuf[0]; inds,_=decode_rle_bitpacked(valbuf[1:],bw,nn); raw=[dictionary[i] for i in inds]
                elif enc==ENC_PLAIN: raw,_=parse_plain(valbuf,ptype,nn,type_length)
                else: raise NotImplementedError(('encodingv2',enc))
                it=iter(raw); values.extend(next(it) if (max_def==0 or d==max_def) else None for d in defs)
            else:
                pass
        assert len(values) == nvalues, (len(values), nvalues, pos, end)
        # logical conversion
        logical=se.get(10,{})
        if ptype==PT_BYTE_ARRAY:
            values=[None if x is None else x.decode('utf-8') for x in values]
        # date logical field 8 (DATE) nested? actual int32 days
        if ptype==PT_INT32 and logical and 8 in logical:
            epoch=datetime.date(1970,1,1); values=[None if x is None else epoch+datetime.timedelta(days=x) for x in values]
        return values

