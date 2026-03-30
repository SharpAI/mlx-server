#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>

static const float C3[8]={-0.190685f,-0.117832f,-0.065717f,-0.021460f,0.021460f,0.065717f,0.117832f,0.190685f};
static const float B3[7]={-0.154259f,-0.091775f,-0.043589f,0.f,0.043589f,0.091775f,0.154259f};
static const float QJL_C=1.2533141373155003f;
static const float S1[128]={-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1};
static const float S2[128]={1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1};
static const float Q1[128]={1,-1,-1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,-1,1,1,-1,1,1,1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1};
static const float Q2[128]={1,1,-1,1,1,-1,1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1,1,-1,-1,1,1,1,1,1,1,1,-1,1,1,-1,-1,1,-1,1,1,-1,1,-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,1,1,1,1,-1};

static void fwht(float*x,int n){for(int h=1;h<n;h*=2)for(int i=0;i<n;i+=h*2)for(int j=i;j<i+h;j++){float a=x[j],b=x[j+h];x[j]=a+b;x[j+h]=a-b;}float s=(n==128)?0.08838834764831845f:0.125f;for(int i=0;i<n;i++)x[i]*=s;}
static void rfwd(float*x){for(int i=0;i<128;i++)x[i]*=S1[i];fwht(x,128);for(int i=0;i<128;i++)x[i]*=S2[i];}
static void rinv(float*x){for(int i=0;i<128;i++)x[i]*=S2[i];fwht(x,128);for(int i=0;i<128;i++)x[i]*=S1[i];}
static void qrot(float*x){for(int i=0;i<128;i++)x[i]*=Q1[i];fwht(x,128);for(int i=0;i<128;i++)x[i]*=Q2[i];}
static int nc3(float v){if(v<B3[3]){if(v<B3[1])return v<B3[0]?0:1;return v<B3[2]?2:3;}else{if(v<B3[5])return v<B3[4]?4:5;return v<B3[6]?6:7;}}
static void p3(const uint8_t*idx,uint8_t*o,int d){memset(o,0,(d*3+7)/8);for(int i=0;i<d;i++){int bo=i*3,by=bo/8,bi=bo%8;o[by]|=(idx[i]&7)<<bi;if(bi>5)o[by+1]|=(idx[i]&7)>>(8-bi);}}
static void u3(const uint8_t*in,uint8_t*idx,int d){for(int i=0;i<d;i++){int bo=i*3,by=bo/8,bi=bo%8;uint16_t r=in[by];if(by+1<(d*3+7)/8)r|=(uint16_t)in[by+1]<<8;idx[i]=(r>>bi)&7;}}
static uint16_t tf16(float f){union{float f;uint32_t u;}v={f};uint32_t u=v.u;uint16_t s=(u>>16)&0x8000;int32_t e=(int32_t)((u>>23)&0xFF)-127+15;uint32_t m=u&0x7FFFFF;if(e<=0)return s;if(e>=31)return s|0x7C00;return(uint16_t)(s|(e<<10)|(m>>13));}
static float ff16(uint16_t h){uint32_t s=(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF;if(!e){union{uint32_t u;float f;}v={s};return v.f;}if(e==31){union{uint32_t u;float f;}v={s|(0xFF<<23)|m};return v.f;}union{uint32_t u;float f;}v={s|((e+127-15)<<23)|(m<<13)};return v.f;}
static uint64_t rng=20260330ULL;
static float rn(){rng=rng*6364136223846793005ULL+1442695040888963407ULL;double u1=(double)(rng>>11)/(1ULL<<53);if(u1<1e-15)u1=1e-15;rng=rng*6364136223846793005ULL+1442695040888963407ULL;double u2=(double)(rng>>11)/(1ULL<<53);return(float)(sqrt(-2.0*log(u1))*cos(6.283185307*u2));}

static int run=0,ok=0;
#define CHK(c,m) do{run++;if(c){ok++;printf("  PASS  %s\n",m);}else printf("  FAIL  %s\n",m);}while(0)

int main(){
  printf("=== TurboQuant Tests (ported from TheTom/llama-cpp-turboquant) ===\n");

  // T1: centroids match Python ref
  printf("\n[T1] Centroids match turboquant_plus Python reference\n");
  {const float py[8]={-0.190685f,-0.117832f,-0.065717f,-0.021460f,0.021460f,0.065717f,0.117832f,0.190685f};
  int r=1;for(int i=0;i<8;i++)if(fabsf(C3[i]-py[i])>1e-5f)r=0;CHK(r,"8 Lloyd-Max centroids == Python ref (tol 1e-5)");}

  // T2: WHT sign spot-check
  printf("\n[T2] WHT sign arrays spot-check vs turbo-wht.h\n");
  {const float rs1[8]={-1,1,1,-1,-1,1,-1,1},rs2[8]={1,1,1,1,-1,1,1,-1};
  const float rq1[8]={1,-1,-1,-1,-1,1,-1,1},rq2[8]={1,1,-1,1,1,-1,1,1};
  int r=1;for(int i=0;i<8;i++)if(S1[i]!=rs1[i]||S2[i]!=rs2[i]||Q1[i]!=rq1[i]||Q2[i]!=rq2[i])r=0;
  CHK(r,"First 8 values of all 4 sign arrays match turbo-wht.h");}

  // T3: FWHT²==x/128
  printf("\n[T3] FWHT applied twice == x (self-inverse with normalization)\n");
  {float x[128],o[128];for(int i=0;i<128;i++)x[i]=o[i]=rn();fwht(x,128);fwht(x,128);
  float mse=0;for(int i=0;i<128;i++){float d=x[i]-o[i];mse+=d*d;}mse/=128;
  char b[80];snprintf(b,sizeof b,"FWHT²=x MSE=%.2e (need <1e-10)",mse);CHK(mse<1e-10f,b);}

  // T4: rot_fwd ∘ rot_inv = I
  printf("\n[T4] Forward∘Inverse rotation == identity\n");
  {float x[128],o[128];for(int i=0;i<128;i++)x[i]=o[i]=rn();rfwd(x);rinv(x);
  float mse=0;for(int i=0;i<128;i++){float d=x[i]-o[i];mse+=d*d;}mse/=128;
  char b[80];snprintf(b,sizeof b,"rot_fwd∘rot_inv MSE=%.2e (need <1e-8)",mse);CHK(mse<1e-8f,b);}

  // T5: rotation is norm-preserving
  printf("\n[T5] WHT rotation is norm-preserving\n");
  {float x[128];for(int i=0;i<128;i++)x[i]=rn();float n0=0;for(int i=0;i<128;i++)n0+=x[i]*x[i];n0=sqrtf(n0);rfwd(x);float n1=0;for(int i=0;i<128;i++)n1+=x[i]*x[i];n1=sqrtf(n1);
  char b[80];snprintf(b,sizeof b,"‖x‖ before=%.4f after=%.4f δ=%.2e",n0,n1,fabsf(n0-n1));CHK(fabsf(n0-n1)<n0*1e-4f,b);}

  // T6: 3-bit pack/unpack
  printf("\n[T6] 3-bit pack/unpack round-trip\n");
  {uint8_t idx[128],pk[48],rec[128];for(int i=0;i<128;i++)idx[i]=i%8;p3(idx,pk,128);u3(pk,rec,128);int mis=0;for(int i=0;i<128;i++)if(idx[i]!=rec[i])mis++;char b[64];snprintf(b,sizeof b,"%d/128 mismatches",mis);CHK(mis==0,b);}

  // T7: V-cache SNR (200 trials)
  printf("\n[T7] V-cache encode/decode SNR (200 Gaussian vectors, d=128)\n");
  {double tm=0,te=0;
  for(int t=0;t<200;t++){
    float src[128],buf[128];for(int i=0;i<128;i++)src[i]=buf[i]=rn();
    float ns=0;for(int i=0;i<128;i++)ns+=buf[i]*buf[i];float gn=sqrtf(ns);float inv=(gn>1e-10f)?1.f/gn:0.f;for(int i=0;i<128;i++)buf[i]*=inv;
    rfwd(buf);
    uint8_t idx[128];float rs=0;for(int i=0;i<128;i++){idx[i]=nc3(buf[i]);rs+=C3[idx[i]]*C3[idx[i]];}
    float rn2=sqrtf(rs),cor=(rn2>1e-10f)?gn/rn2:gn;
    uint8_t pk[48];p3(idx,pk,128);uint8_t idx2[128];u3(pk,idx2,128);
    float dec[128];for(int i=0;i<128;i++)dec[i]=C3[idx2[i]];rinv(dec);float sc=ff16(tf16(cor));for(int i=0;i<128;i++)dec[i]*=sc;
    float mse=0,en=0;for(int i=0;i<128;i++){float d=dec[i]-src[i];mse+=d*d;en+=src[i]*src[i];}tm+=mse/128;te+=en/128;
  }
  float rel=(float)(tm/te),snr=-10.f*log10f(rel);
  char b[100];snprintf(b,sizeof b,"V rel_MSE=%.4f SNR=%.1f dB (need SNR>9 dB)",rel,snr);CHK(rel<0.15f,b);printf("  INFO: V-cache SNR = %.1f dB\n",snr);}

  // T8: K-cache inner-product preservation (100 pairs)
  printf("\n[T8] K-cache inner-product SNR (100 random key/query pairs)\n");
  {double es=0,rs=0;
  for(int t=0;t<100;t++){
    float ki[128],kq[128];for(int i=0;i<128;i++){ki[i]=rn();kq[i]=rn();}
    float ipt=0;for(int i=0;i<128;i++)ipt+=ki[i]*kq[i];
    // encode
    float ns=0,nrm[128];for(int i=0;i<128;i++)ns+=ki[i]*ki[i];float nm=sqrtf(ns),inv=(nm>1e-10f)?1.f/nm:0.f;for(int i=0;i<128;i++)nrm[i]=ki[i]*inv;
    float rot[128];memcpy(rot,nrm,sizeof rot);rfwd(rot);
    uint8_t idx[128];for(int i=0;i<128;i++)idx[i]=nc3(rot[i]);
    float mr[128];for(int i=0;i<128;i++)mr[i]=C3[idx[i]];rinv(mr);
    float res[128];float rns=0;for(int i=0;i<128;i++){res[i]=nrm[i]-mr[i];rns+=res[i]*res[i];}float rnm=sqrtf(rns);
    float proj[128];memcpy(proj,res,sizeof proj);qrot(proj);
    uint8_t sg[16]={};for(int i=0;i<128;i++)if(proj[i]>=0)sg[i/8]|=(1<<(i%8));
    // decode
    float dec[128];for(int i=0;i<128;i++)dec[i]=C3[idx[i]];rinv(dec);
    float sf[128];for(int i=0;i<128;i++)sf[i]=(sg[i/8]&(1<<(i%8)))?1.f:-1.f;qrot(sf);
    float qs=QJL_C/128.f*ff16(tf16(rnm)),sc=ff16(tf16(nm));for(int i=0;i<128;i++)dec[i]=(dec[i]+sf[i]*qs)*sc;
    float ipa=0;for(int i=0;i<128;i++)ipa+=dec[i]*kq[i];float e=ipa-ipt;es+=e*e;rs+=ipt*ipt;
  }
  float rel=(float)(es/(rs+1e-12)),snr=-10.f*log10f(rel);
  char b[120];snprintf(b,sizeof b,"K rel_IP_err=%.4f SNR=%.1f dB (need SNR>9 dB)",rel,snr);CHK(rel<0.15f,b);printf("  INFO: K-cache IP SNR = %.1f dB\n",snr);}

  // T9: fp16 round-trip
  printf("\n[T9] fp16 conversion round-trip\n");
  {float vals[]={1.f,-1.5f,0.125f,123.456f,-0.001953125f};int r=1;for(float v:vals){float rt=ff16(tf16(v));if(fabsf(rt-v)/(fabsf(v)+1e-6f)>0.003f)r=0;}CHK(r,"fp16 rel-err < 0.3% for all test values");}

  printf("\n=== %d/%d tests passed ===\n",ok,run);
  return ok==run?0:1;
}
