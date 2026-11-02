import numpy as np, glob, sqlite3, os, random
random.seed(0)
root=os.path.expanduser("~/nuplan/labels/train-labels-2k")
dbs={os.path.basename(p)[:-3]:p for p in glob.glob(os.path.expanduser("~/nuplan/dataset/nuplan-v1.1/*/*.db"))}
files=sorted(glob.glob(root+"/*/*.npz")); random.shuffle(files)
rows=[]
for f in files:
    if len(rows)>=600: break
    z=np.load(f)
    log=str(z["log_name"]); tok=str(z["scenario_token"]); it=int(z["iteration"])
    if log not in dbs: continue
    con=sqlite3.connect(dbs[log])
    r=con.execute("select timestamp from lidar_pc where token=?",(bytes.fromhex(tok),)).fetchone()
    if r is None: con.close(); continue
    t=r[0]+it*100000
    e=con.execute("select vx,vy,acceleration_x,acceleration_y from ego_pose order by abs(timestamp-?) limit 1",(t,)).fetchone()
    con.close()
    c=z["candidates"]                      # [K,80,4] ego frame (x,y,cos,sin)
    v_imp=np.linalg.norm(c[:,0,:2],axis=-1)/0.1   # implied speed from first step
    v_imp5=np.linalg.norm(c[:,4,:2],axis=-1)/0.5
    rows.append((np.hypot(e[0],e[1]), v_imp.mean(), v_imp.std(), v_imp5.mean(), np.hypot(e[2],e[3])))
a=np.array(rows); print("n",len(a))
v=a[:,0]
for name,i in [("first step (0.1s)",1),("5 steps (0.5s)",3)]:
    r=np.corrcoef(v,a[:,i])[0,1]; print(name,"corr",round(r,3),"R2",round(r*r,3),"MAE m/s",round(np.abs(v-a[:,i]).mean(),2))
print("true speed mean/std",v.mean().round(2),v.std().round(2),"| frac stopped (<0.2)",(v<0.2).mean().round(2))
print("implied-speed spread across K (mean std)",a[:,2].mean().round(3))
