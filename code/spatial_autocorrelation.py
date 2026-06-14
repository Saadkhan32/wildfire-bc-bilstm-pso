import warnings, csv, math, statistics as st, numpy as np, pyproj
warnings.filterwarnings("ignore")
from libpysal.weights import KNN, DistanceBand, Kernel
import esda
from scipy.spatial import cKDTree
TF=pyproj.Transformer.from_crs(4326,3005,always_xy=True)
SEEDS=[42,101,202,303,404,505,606,707,808,909]
def load(thr,seed):
    rows=list(csv.DictReader(open(f"data/training_points_{thr}ha_seed{seed}.csv")))
    lon=np.array([float(r["Longitude"]) for r in rows]); lat=np.array([float(r["Latitude"]) for r in rows])
    y=np.array([int(r["Status"]) for r in rows],float); X,Y=TF.transform(lon,lat); return np.c_[X,Y],y
def indices(coords,y):
    w=KNN.from_array(coords,k=8); w.transform="r"
    mi=esda.Moran(y,w,permutations=0); gc=esda.Geary(y,KNN.from_array(coords,k=8),permutations=0)
    gg=esda.G(y,KNN.from_array(coords,k=8),permutations=0); jc=esda.Join_Counts(y.astype(int),KNN.from_array(coords,k=8),permutations=0)
    return mi.I,mi.z_norm,gc.C,gg.z_norm,jc.bb
print("seed,MoranI,Moran_z,GearyC,GetisG_z,JoinBB")
agg={k:[] for k in "I C G B".split()}
for s in SEEDS:
    c,y=load(70,s); I,z,C,Gz,BB=indices(c,y); print(f"{s},{I:.4f},{z:.1f},{C:.4f},{Gz:.1f},{BB:.0f}")
    agg["I"].append(I);agg["C"].append(C);agg["G"].append(Gz);agg["B"].append(BB)
print("\nMEAN+/-SD across seeds:")
for k,n in [("I","Moran I"),("C","Geary C"),("G","Getis z"),("B","Join BB")]:
    v=agg[k]; print(f"  {n}: {st.mean(v):.3f} +/- {st.stdev(v):.3f}")
c,y=load(70,42)
for nm,w in [("KNN8",KNN.from_array(c,k=8)),("band50km",DistanceBand.from_array(c,threshold=50000,binary=True,silence_warnings=True)),("kernel",Kernel.from_array(c,fixed=False,k=15,function="triangular"))]:
    w.transform="r"; print(f"Moran ({nm}) = {esda.Moran(y,w,permutations=0).I:.3f}")
fire=c[y==1]; d,_=cKDTree(fire).query(fire,k=2); nnd=d[:,1].mean()
area=(c[:,0].max()-c[:,0].min())*(c[:,1].max()-c[:,1].min()); exp=0.5/math.sqrt(len(fire)/area)
print(f"ANN ratio (fire) = {nnd/exp:.3f}")
