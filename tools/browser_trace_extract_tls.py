"""Extract TLS byte streams from a local SOCKS5 capture for offline Wireshark analysis.

The output PCAP reconstructs TCP/IP headers and omits SOCKS negotiation. TLS payload bytes
are unchanged. It is not a TCP fingerprint/timing reference. Original PCAPs and outputs
must stay private, especially when combined with a TLS key log.

Requires dpkt==1.9.8. Capture format: Ethernet/IPv4 PCAP, SOCKS5 with no authentication.
Example:
  python3 tools/browser_trace_extract_tls.py raw.pcap --socks-port 42451 --output-dir private/extracted
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import dpkt

DEFAULT_HOSTS = ['accounts.google.com', 'accounts.youtube.com', 'www.google.com',
                 'www.gstatic.com', 'fonts.gstatic.com', 'play.google.com']


def extract(source, port, output_dir, hosts=DEFAULT_HOSTS):
    ROOT = output_dir
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    active={};flows=[];events=[]
    with source.open('rb') as f:
     for ts,raw in dpkt.pcap.Reader(f):
      eth=dpkt.ethernet.Ethernet(raw)
      if not isinstance(eth.data,dpkt.ip.IP) or not isinstance(eth.data.data,dpkt.tcp.TCP):continue
      tcp=eth.data.data;client=tcp.dport==port;cp=tcp.sport if client else tcp.dport
      if tcp.sport!=port and tcp.dport!=port:continue
      if client and tcp.flags&dpkt.tcp.TH_SYN and not tcp.flags&dpkt.tcp.TH_ACK:
       flow={'id':len(flows),'port':cp,'isn':[tcp.seq,None],'parts':[[],[]]};flows.append(flow);active[cp]=flow
      flow=active.get(cp)
      if flow is None:continue
      direction=0 if client else 1
      if not client and tcp.flags&dpkt.tcp.TH_SYN:flow['isn'][1]=tcp.seq
      if tcp.data and flow['isn'][direction] is not None:
       off=(tcp.seq-flow['isn'][direction]-1)&0xffffffff
       assert off<128*1024*1024
       data=bytes(tcp.data);flow['parts'][direction].append((off,data));events.append((ts,flow,direction,off,data,tcp.ack))

    def reassemble(parts):
     if not parts:return b''
     out=bytearray();end=0
     for off,data in sorted(parts,key=lambda p:p[0]):
      if off>end:raise ValueError('Missing TCP bytes')
      overlap=min(max(0,end-off),len(data))
      if overlap and out[off:off+overlap]!=data[:overlap]:raise ValueError('Conflicting TCP retransmission')
      if off+len(data)>end:out.extend(data[max(0,end-off):]);end=off+len(data)
     return bytes(out)
    def address(data,pos):
     atyp=data[pos];pos+=1
     if atyp==3:n=data[pos];pos+=1;host=data[pos:pos+n].decode();pos+=n
     elif atyp==1:host=socket.inet_ntop(socket.AF_INET,data[pos:pos+4]);pos+=4
     elif atyp==4:host=socket.inet_ntop(socket.AF_INET6,data[pos:pos+16]);pos+=16
     else:raise ValueError('Invalid SOCKS address')
     return host,int.from_bytes(data[pos:pos+2],'big'),pos+2
    approved=set(hosts)
    selected=[]
    for flow in flows:
     try:
      c,s=map(reassemble,flow['parts']);assert c[:1]==b'\x05' and s[:2]==b'\x05\x00'
      request=2+c[1];assert c[request:request+3]==b'\x05\x01\x00'
      host,dest,coff=address(c,request+3);_,_,soff=address(s,5)
      if host not in approved or dest!=443:continue
      assert s[2:5]==b'\x05\x00\x00'
      assert c[coff:coff+1]==b'\x16'
      flow.update(host=host,skip=[coff,soff],streams=[c[coff:],s[soff:]],output_port=30000+len(selected),started=False)
      selected.append(flow)
     except (ValueError,AssertionError,IndexError):continue

    def packet(flow,direction,seq,ack,flags,data=b''):
     a,b=('198.18.0.1','198.18.0.2') if direction==0 else ('198.18.0.2','198.18.0.1')
     sport,dport=(flow['output_port'],443) if direction==0 else (443,flow['output_port'])
     tcp=dpkt.tcp.TCP(sport=sport,dport=dport,seq=seq,ack=ack,flags=flags,win=65535,data=data);tcp.off=5
     ip=dpkt.ip.IP(src=socket.inet_aton(a),dst=socket.inet_aton(b),p=6,ttl=64,data=tcp);ip.len=len(ip)
     return bytes(dpkt.ethernet.Ethernet(src=b'\x02\0\0\0\0\x01',dst=b'\x02\0\0\0\0\x02',type=dpkt.ethernet.ETH_TYPE_IP,data=ip))
    with (ROOT/'tls-only.pcap').open('wb') as f:
     writer=dpkt.pcap.Writer(f)
     for ts,flow,direction,off,data,ack in events:
      if 'host' not in flow:continue
      offset=off-flow['skip'][direction]
      if offset+len(data)<=0:continue
      if offset<0:data=data[-offset:];offset=0
      if not flow['started']:
       writer.writepkt(packet(flow,0,0,0,dpkt.tcp.TH_SYN),ts-.000003)
       writer.writepkt(packet(flow,1,0,1,dpkt.tcp.TH_SYN|dpkt.tcp.TH_ACK),ts-.000002)
       writer.writepkt(packet(flow,0,1,1,dpkt.tcp.TH_ACK),ts-.000001);flow['started']=True
      other=1-direction;other_ack=max(1,((ack-flow['isn'][other]-1)&0xffffffff)-flow['skip'][other]+1)
      # Split offloaded loopback segments to valid IPv4 packet lengths. TLS bytes stay unchanged.
      for i in range(0,len(data),60000):writer.writepkt(packet(flow,direction,1+offset+i,other_ack,dpkt.tcp.TH_ACK|dpkt.tcp.TH_PUSH,data[i:i+60000]),ts+i*1e-12)
    manifest=[]
    for flow in selected:
     manifest.append({'original_flow':flow['id'],'output_port':flow['output_port'],'host':flow['host'],'tls_lengths':[len(x) for x in flow['streams']],'tls_sha256':[hashlib.sha256(x).hexdigest() for x in flow['streams']]})
    (ROOT/'tls-only-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pcap', type=Path)
    parser.add_argument('--socks-port', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--host', action='append', help='Include this hostname instead of the default Google hosts')
    args = parser.parse_args()
    os.umask(0o077)
    manifest = extract(args.pcap, args.socks_port, args.output_dir, args.host or DEFAULT_HOSTS)
    print(f'Extracted {len(manifest)} TLS flows. TCP/IP metadata is reconstructed; TLS bytes are unchanged.')
