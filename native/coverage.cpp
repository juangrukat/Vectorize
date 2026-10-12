#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
struct Derivative { int id; std::array<double,4> j; };
struct Point { double x,y; std::vector<Derivative> d; };
static void add(Point &p,int id,const std::array<double,4>&j) {
 for(auto &d:p.d) if(d.id==id){for(int k=0;k<4;k++)d.j[k]+=j[k];return;}
 p.d.push_back({id,j});
}
static Point intersection(const Point&a,const Point&b,int axis,double bound,bool grad){
 double av=axis?a.y:a.x, bv=axis?b.y:b.x, den=bv-av;
 double t=(bound-av)/den;
 Point p{a.x+t*(b.x-a.x),a.y+t*(b.y-a.y),{}};
 if(grad){
  for(const auto &d:a.d){std::array<double,4> j{};for(int k=0;k<2;k++){
   double dt=-(1-t)*d.j[2*axis+k]/den;
   j[k]=(1-t)*d.j[k]+(b.x-a.x)*dt;
   j[2+k]=(1-t)*d.j[2+k]+(b.y-a.y)*dt;
  }add(p,d.id,j);}
  for(const auto &d:b.d){std::array<double,4> j{};for(int k=0;k<2;k++){
   double dt=-t*d.j[2*axis+k]/den;
   j[k]=t*d.j[k]+(b.x-a.x)*dt;
   j[2+k]=t*d.j[2+k]+(b.y-a.y)*dt;
  }add(p,d.id,j);}
 }
 if(axis)p.y=bound;else p.x=bound;
 return p;
}
static std::vector<Point> clip(const std::vector<Point>&in,int axis,double bound,bool lower,bool grad){
 std::vector<Point> out;if(in.empty())return out;out.reserve(in.size()+2);
 auto inside=[&](const Point&p){double v=axis?p.y:p.x;return lower?v>=bound:v<=bound;};
 const Point *a=&in.back();bool ia=inside(*a);
 for(const auto &b:in){bool ib=inside(b);if(ia!=ib)out.push_back(intersection(*a,b,axis,bound,grad));if(ib)out.push_back(b);a=&b;ia=ib;}
 return out;
}
static double area(const std::vector<Point>&p){double a=0;if(p.empty())return a;const Point *prev=&p.back();for(const auto&v:p){a+=prev->x*v.y-v.x*prev->y;prev=&v;}return a*.5;}
struct Input {
 const double *v,*pal;int n,k,w,h;std::vector<std::vector<int>> loops;std::vector<int> colors;
 Input(const Array&vertices,std::vector<std::vector<int>> ls,std::vector<int>cs,const Array&palette,int width,int height):w(width),h(height),loops(std::move(ls)),colors(std::move(cs)){
  if(vertices.ndim()!=2||vertices.shape(1)!=2||vertices.shape(0)>1000000)throw std::invalid_argument("vertices must be N x 2 (at most 1000000)");
  if(palette.ndim()!=2||palette.shape(1)!=4||palette.shape(0)<1||palette.shape(0)>4096)throw std::invalid_argument("palette must be K x 4 with 1 <= K <= 4096");
  if(w<=0||h<=0||w>16384||h>16384||static_cast<long long>(w)*h>16777216)throw std::invalid_argument("invalid or excessive image dimensions");
  n=static_cast<int>(vertices.shape(0));k=static_cast<int>(palette.shape(0));v=vertices.data();pal=palette.data();
  if(loops.size()!=colors.size()||loops.size()>1000000)throw std::invalid_argument("loop_colors must match loops");
  for(int i=0;i<n*2;i++)if(!std::isfinite(v[i])||std::abs(v[i])>1e7)throw std::invalid_argument("vertices must be finite and bounded");
  for(int i=0;i<k*4;i++)if(!std::isfinite(pal[i]))throw std::invalid_argument("palette must be finite");
  size_t total=0;for(size_t i=0;i<loops.size();i++){if(loops[i].size()<3||colors[i]<0||colors[i]>=k)throw std::invalid_argument("invalid loop or color");for(int id:loops[i])if(id<0||id>=n)throw std::invalid_argument("vertex index out of bounds");total+=loops[i].size();}if(total>4000000)throw std::invalid_argument("excessive loop complexity");
 }
 template<class F>void visit(bool grad,F fn)const{
  for(size_t l=0;l<loops.size();l++){
   std::vector<Point> p;double xmin=w,xmax=0,ymin=h,ymax=0;
   for(int id:loops[l]){Point q{v[id*2],v[id*2+1],{}};if(grad)q.d.push_back({id,{1,0,0,1}});p.push_back(std::move(q));xmin=std::min(xmin,v[id*2]);xmax=std::max(xmax,v[id*2]);ymin=std::min(ymin,v[id*2+1]);ymax=std::max(ymax,v[id*2+1]);}
   int x0=std::max(0,static_cast<int>(std::floor(xmin))),x1=std::min(w,static_cast<int>(std::ceil(xmax)));
   int y0=std::max(0,static_cast<int>(std::floor(ymin))),y1=std::min(h,static_cast<int>(std::ceil(ymax)));
   // Clip the full loop once to each horizontal pixel slab. The remaining
   // per-pixel work depends on the local boundary complexity, not all vertices.
   // Signed shoelace integration cancels doubled connectors for disconnected
   // intersections of a concave polygon with the slab.
   for(int y=y0;y<y1;y++){
    auto slab=clip(p,1,y,true,grad);
    slab=clip(slab,1,y+1,false,grad);
    if(slab.size()<3)continue;
    double row_min=w,row_max=0;
    for(const auto &q:slab){row_min=std::min(row_min,q.x);row_max=std::max(row_max,q.x);}
    int row_x0=std::max(x0,static_cast<int>(std::floor(row_min)));
    int row_x1=std::min(x1,static_cast<int>(std::ceil(row_max)));
    for(int x=row_x0;x<row_x1;x++){
     auto q=clip(slab,0,x,true,grad);q=clip(q,0,x+1,false,grad);
     if(q.size()>=3)fn(q,colors[l],y*w+x);
    }
   }
  }
 }
 void render(double*out)const{visit(false,[&](const auto&q,int c,int idx){double a=area(q);for(int j=0;j<4;j++)out[idx*4+j]+=a*pal[c*4+j];});}
};
PYBIND11_MODULE(_coverage,m){
 m.doc()="Exact signed polygon pixel coverage and analytic clipping derivatives.";
 m.def("render",[](Array vertices,std::vector<std::vector<int>> loops,std::vector<int>colors,Array palette,int w,int h){
  Input in(vertices,std::move(loops),std::move(colors),palette,w,h);Array out({h,w,4});std::fill(out.mutable_data(),out.mutable_data()+h*w*4,0.0);
  {py::gil_scoped_release release;in.render(out.mutable_data());}return out;
 },py::arg("vertices"),py::arg("loops"),py::arg("loop_colors"),py::arg("palette"),py::arg("width"),py::arg("height"));
 m.def("loss_gradient",[](Array vertices,std::vector<std::vector<int>> loops,std::vector<int>colors,Array palette,int w,int h,Array target){
  Input in(vertices,std::move(loops),std::move(colors),palette,w,h);
  if(target.ndim()!=3||target.shape(0)!=h||target.shape(1)!=w||target.shape(2)!=4)throw std::invalid_argument("target must be height x width x 4");
  for(ssize_t i=0;i<target.size();i++)if(!std::isfinite(target.data()[i]))throw std::invalid_argument("target must be finite");
  Array gradient({in.n,2});std::fill(gradient.mutable_data(),gradient.mutable_data()+in.n*2,0.0);double loss=0;
  {py::gil_scoped_release release;std::vector<double> residual(static_cast<size_t>(h)*w*4,0);in.render(residual.data());
   for(size_t i=0;i<residual.size();i++){residual[i]-=target.data()[i];loss+=.5*residual[i]*residual[i];}
   in.visit(true,[&](const auto&q,int c,int idx){double factor=0;for(int j=0;j<4;j++)factor+=residual[idx*4+j]*in.pal[c*4+j];
    for(size_t i=0;i<q.size();i++){const auto&a=q[(i+q.size()-1)%q.size()];const auto&b=q[(i+1)%q.size()];double gx=.5*(b.y-a.y),gy=.5*(a.x-b.x);
     for(const auto&d:q[i].d)for(int j=0;j<2;j++)gradient.mutable_data()[d.id*2+j]+=factor*(gx*d.j[j]+gy*d.j[2+j]);}
   });
  }return py::make_tuple(loss,gradient);
 },py::arg("vertices"),py::arg("loops"),py::arg("loop_colors"),py::arg("palette"),py::arg("width"),py::arg("height"),py::arg("target"));
}
