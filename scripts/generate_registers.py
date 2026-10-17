"""Extract the real HLS control ABI; do not guess register offsets."""
from pathlib import Path
import json
import re

root=Path(__file__).resolve().parents[1]
header=root/'build/hls/solution/impl/misc/drivers/bosio_projector_v1_0/src/xbosio_projector_hw.h'
macros=dict(re.findall(r'#define XBOSIO_PROJECTOR_CONTROL_ADDR_(\w+)\s+(0x[0-9a-f]+)',header.read_text()))
regs={'control':int(macros['AP_CTRL'],16),'return':int(macros['AP_RETURN'],16)}
for name in ('texture','descriptors','scene','texture_words','descriptor_words','scene_words','job_count','written_cells'):
    regs[name]=int(macros[name.upper()+'_DATA'],16)
(root/'sw/registers.json').write_text(json.dumps(regs,indent=2)+'\n')
print(json.dumps(regs))
