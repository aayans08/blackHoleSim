import math, datetime
M32=0xFFFFFFFF
def pcg3d(x,y,z):
    x=(x*1664525+1013904223)&M32; y=(y*1664525+1013904223)&M32; z=(z*1664525+1013904223)&M32
    x=(x+y*z)&M32; y=(y+z*x)&M32; z=(z+x*y)&M32
    x^=x>>16; y^=y>>16; z^=z>>16
    x=(x+y*z)&M32; y=(y+z*x)&M32; z=(z+x*y)&M32
    return x,y,z
for v in [(0,0,0),(1,2,3),(0xFFFFFFFF,7,42)]:
    print('pcg3d',v,[hex(a) for a in pcg3d(*v)])

# Binet Cartesian RK4 in units M=1 (Rs=2). ray from far away with impact parameter b (in M), deflection angle
def deflect(b, x0=-1e4, h0=None):
    x=[x0, b, 0.0]; v=[1.0,0.0,0.0]
    def acc(p,vv):
        r=math.sqrt(sum(c*c for c in p))
        hx=[p[1]*vv[2]-p[2]*vv[1], p[2]*vv[0]-p[0]*vv[2], p[0]*vv[1]-p[1]*vv[0]]
        h2=sum(c*c for c in hx)
        k=-3.0*h2/r**5
        return [k*c for c in p]
    steps=0
    while True:
        r=math.sqrt(sum(c*c for c in x))
        if r<2.0*(1+1e-6): return None
        if x[0]>-x0 and (x[0]*v[0]+x[1]*v[1])>0: break
        dl=max(1e-3, 0.01*r)
        k1v=acc(x,v); k1x=v
        x2=[x[i]+0.5*dl*k1x[i] for i in range(3)]; v2=[v[i]+0.5*dl*k1v[i] for i in range(3)]
        k2v=acc(x2,v2); k2x=v2
        x3=[x[i]+0.5*dl*k2x[i] for i in range(3)]; v3=[v[i]+0.5*dl*k2v[i] for i in range(3)]
        k3v=acc(x3,v3); k3x=v3
        x4=[x[i]+dl*k3x[i] for i in range(3)]; v4=[v[i]+dl*k3v[i] for i in range(3)]
        k4v=acc(x4,v4); k4x=v4
        x=[x[i]+dl/6*(k1x[i]+2*k2x[i]+2*k3x[i]+k4x[i]) for i in range(3)]
        v=[v[i]+dl/6*(k1v[i]+2*k2v[i]+2*k3v[i]+k4v[i]) for i in range(3)]
        steps+=1
    ang=math.atan2(-v[1],v[0])
    return ang
for bRs in [5,10,50]:
    b=bRs*2.0
    a=deflect(b)
    # correct for finite start/end distance: asymptotic weak-field residual 2*(2M/x0) approx; use large x0
    print('deflection b=%g Rs: %.5f rad'%(bRs,a))
bc=3*math.sqrt(3)
print('b_c = %.6f M = %.6f Rs'%(bc,bc/2))
print('b=2.60 Rs captured?', deflect(2.60*2) is None, ' b=2.597 Rs captured?', deflect(2.597*2) is None)
# shadow angular radius static observer r=30M
r=30.0
s=bc/r*math.sqrt(1-2/r); th=math.asin(s)
print('shadow theta deg %.4f'%math.degrees(th))
H=1080; fov=math.radians(60)
print('shadow radius px (1920x1080,fovy60) %.3f'%((H/2)*math.tan(th)/math.tan(fov/2)))
# star illuminance
lux=lambda m:10**(-0.4*(m+13.99))
print('lux m0 %.4e  sun %.4e'%(lux(0),lux(-26.74)))
bv=0.65; T=4600*(1/(0.92*bv+1.7)+1/(0.92*bv+0.62)); print('Teff(B-V=0.65)=%.1f'%T)
# epoch: 2026-11-20T00:00:00Z -> TDB seconds since J2000 (TAI-UTC=37, TT=TAI+32.184, TDB~TT)
j2000_utc=datetime.datetime(2000,1,1,11,58,55,816000,tzinfo=datetime.timezone.utc)
d=datetime.datetime(2026,11,20,0,0,0,tzinfo=datetime.timezone.utc)
dt=(d-j2000_utc).total_seconds()  # UTC elapsed (no leap secs)
leaps_since_2000 = 37-32
print('unix J2000 UTC', j2000_utc.timestamp())
print('2026-11-20T00Z TDB sec since J2000 = %.3f (day %d, sec %.3f)'%(dt+leaps_since_2000, (dt+leaps_since_2000)//86400, (dt+leaps_since_2000)%86400))
# galactic center direction check
A=[[-0.0548755604162154,-0.8734370902348850,-0.4838350155487132],[0.4941094278755837,-0.4448296299600112,0.7469822444972189],[-0.8676661490190047,-0.1980763734312015,0.4559837761750669]]
gx=A[0]  # galactic x axis expressed in ICRS
ra=math.degrees(math.atan2(gx[1],gx[0]))%360; dec=math.degrees(math.asin(gx[2]))
print('GC RA %.4f Dec %.4f'%(ra,dec))
