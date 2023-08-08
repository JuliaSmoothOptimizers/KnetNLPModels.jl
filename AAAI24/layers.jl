using Knet

# Convolutional layer
struct Conv; w; b; f; pool_option; end
(c::Conv)(x) = c.f.(pool(conv4(c.w, x) .+ c.b; mode=c.pool_option))
Conv(w1::Int, w2::Int, cx::Int, cy::Int, f::Function=sigm; pool_option::Int=0) = Conv(param(w1, w2, cx, cy), param0(1,1, cy,1), f, pool_option)

# Dense layer
struct Dense; w; b; f; p; end
(d::Dense)(x) = d.f.(d.w * mat(dropout(x,d.p)) .+ d.b)
size(d::Dense) = size(d.w)
Dense(i::Int, o::Int, f=sigm;pdrop=0.) = Dense(param(o,i), param0(o), f, pdrop)

# Separable layer
struct Sep_layer; vec_wi; vec_bi; N; f; gcdi; gcdo; in; out; p; end
function (psd::Sep_layer)(x)   
  mapreduce( (j -> psd.f.( psd.vec_wi[j]*mat(dropout(x,psd.p))[(psd.gcdi*(j-1)+1):(psd.gcdi*j),:] .+ (psd.vec_bi[j]) ) ), ((x,y)->vcat(x,y)), [1:psd.N;] )
end
Sep_layer(i::Int, o::Int; N::Int=gcd(i, o), f=sigm,p=0.) = Sep_layer( map((j->param(checkfloor(N, o,i)...)),[1:N;]), map((x->param0(checkfloor(N, o))),[1:N;]), N, f , checkfloor(N,i), checkfloor(N, o), checkfloor(N,i)*N, checkfloor(N, o)*N, p)
SL(i::Int, N::Int, ni::Int; f=sigm) = Sep_layer(i, N*ni;N=N, f=f)

checkfloor(N, elt) = (Int)(max(floor(elt/N), 1)) 
checkfloor(N, e1, e2) = Vector{Int}([checkfloor(N, e1), checkfloor(N, e2)])


