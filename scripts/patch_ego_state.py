import numpy as np, glob, sqlite3, os, json, sys
from pathlib import Path
root=Path(os.path.expanduser("~/nuplan/labels/train-labels-2k")); out=Path(os.path.expanduser("~/nuplan/labels/train-labels-2k-ego"))
n=json.load(open(os.path.expanduser("~/IDL_project/diffusion-planner-scorer/normalization.json")))["ego_current_state"]
mean=np.array(n["mean"],dtype=np.float32); std=np.array(n["std"],dtype=np.float32)
dbs={os.path.basename(p)[:-3]:p for p in glob.glob(os.path.expanduser("~/nuplan/dataset/nuplan-v1.1/*/*.db"))}
WB=3.089
cache={}
def conn(log):
    if log not in cache: cache[log]=sqlite3.connect(dbs[log])
    return cache[log]
miss=0; tot=0
for f in sorted(root.glob("*/*.npz")):
    z=dict(np.load(f)); tot+=1
    log=str(z["log_name"]); tok=str(z["scenario_token"]); it=int(z["iteration"])
    con=conn(log)
    r=con.execute("select timestamp from lidar_pc where token=?",(bytes.fromhex(tok),)).fetchone()
    if r is None: miss+=1; continue
    t=r[0]+it*100000
    e=con.execute("select vx,vy,acceleration_x,acceleration_y,angular_rate_z from ego_pose order by abs(timestamp-?) limit 1",(t,)).fetchone()
    vx,vy,ax,ay,yr=e
    v=vx
    if abs(v)<0.2: steer=0.0; yr=0.0
    else:
        steer=float(np.clip(np.arctan(yr*WB/abs(v)),-2/3*np.pi,2/3*np.pi)); yr=float(np.clip(yr,-0.95,0.95))
    raw=np.array([0,0,1,0,vx,vy,ax,ay,steer,yr],dtype=np.float32)
    z["input_ego_current_state"]=(raw-mean)/std
    d=out/f.parent.name; d.mkdir(parents=True,exist_ok=True)
    np.savez(d/f.name,**z)
print("done",tot,"missing",miss)
