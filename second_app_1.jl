using Convex, SCS, MLDatasets, LinearAlgebra

using InvertedIndices

train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

a1 = permutedims(train_x, [2,1,3])
a2 = transpose(reshape(a1, (784, 60000)))
X=hcat(ones(60000),a2)


C = zeros(length(train_y), 10)

for i in 1:length(train_y)
  C[i, train_y[i]+1]=1
end

Nval = 1000;
X_val = X[1:Nval,:] 
Y_val = C[1:Nval, :]
X_tr = X[1001:60000,:] 
Y_tr = C[1001:60000, :]
q,m=size(X_tr)


ind = [1]
L = ones(q, 1)

for i in 1:m
    v = count(x -> x>0, X_tr[:,i])
    if v<600
        ind = hcat(ind,[i])

    end
            
end
ind = ind[2:end]

L = X_tr[:, Not(ind)]
L2 = X_val[:, Not(ind)]

q , fe = size(L)

randmat = [rand() < 0.5 for _ in 1:5000, _ in 1:fe]
randmat = round.(Int, randmat)
randmat[randmat .== 0] .= -1

T = L*transpose(randmat)
T2 = L2*transpose(randmat)
T[T .< 0] .= 0
T2[T2 .< 0] .= 0
Xnew = [L T]
X2new = [L2 T2]

X_tr = Xnew
X_val = X2new

q , m = size(X_tr)

btest1 = permutedims(test_x, [2,1,3])
btest2 = transpose(reshape(btest1, (784, 10000)))
X_2= hcat(ones(10000),btest2)

n_iter=5

#row_val = rand(1:60000, 1000)


#row_tr = Not(rwo_val)

  #A_tr=vcat(X_tr ,exp(w)*I(m))
  #A_val=X_val


function theta_ls(w, B)
    theta=Variable(m)
    objective =norm(vcat(X_tr,exp(w)*I(m))*theta-B)
    problem = minimize(objective)
    solve!(problem, SCS.Optimizer,silent_solver = true)
    return(Matrix(theta.value))
end

function theta_ls2(w, C)
   A = vcat(X_tr ,exp(w)*I(m));
   return(A\C)
end

function Yh_val(theta)
    return(X_val*theta)
end

function DAT(X_tr,w)
      v, m = size(X_tr)
      O = zeros(v, m)
      return(hcat(transpose(O),exp(w)*I(m)))
end

#A^T*A=L
function DL(X_tr, w)
      v, m = size(X_tr)
      return(2*exp(w)*I(m))
end

function Dtheta(A,B, w)
    L = transpose(A)*A
    return(inv(L)*(DAT(X_tr,w).-DL(X_tr, w)*(inv(L)*transpose(A)))*B)

end

function Loss(Y1, Y2)
   e = findall(x -> x>0, Y2)
  return(-Y1[e].+log(sum(exp.(Y1))))
end

#y1 => A*Dtheta
#y2 => yhat
#y3 => ytrain
function DLoss(y1, y2, y3)
    e = findall(x -> x>0, y3)
    return(-y1[e].+(1/(sum(exp.(y2))))*(transpose(exp.(y2))*y1))
end

function g(theta,w,B)
    A = Matrix(vcat(X_tr ,exp(w)*I(m)));
    y1 = X_val*Dtheta(A,B, w);
    Dsi =0;
    Yh = Yh_val(theta)
    r1 , r2 = size(Y_val)
    for i in 1:r1
    Dsi = Dsi .+ DLoss(y1[i,:],Yh[i,:] , Y_val[i,:])
    end
    return(1/Nval*Dsi)
end

function F(Y_val, theta)
    si = 0
    Yh = Yh_val(theta)
    r1 , r2 = size(Y_val)
    for i in 1:r1
    si = si .+(Loss(Yh[i,:], Y_val[i,:]))
    end
    return(1/Nval*si)
end

function iterr(n_iter, w, B)
    t=1;
    eps=.01;
    for i in 1
        temp=theta_ls2(w, B)
        w_tent=w.-t*g(temp,w,B)
        w_tent = w_tent[1]
        if F(Y_val,theta_ls2(w_tent, B))<=F(Y_val,temp)
            t=1.2*t
            temp2=w
            w=w_tent
            if norm((temp2-w)/t.+g(temp,w,B).-g(temp,temp2,B))<=eps
                break
            end
        else
            t=t/2
        end
    end
    return(w)
end

v,m = size(X_tr)

  w=-2;
  A = Matrix(vcat(X_tr ,exp(w)*I(m)));
  Btrain = vcat(Y_tr, zeros(m,10));
  w = iterr(n_iter, w, Btrain);
  beta=theta_ls2(w, Btrain);


beta

function myreduction(X2, ind)
p , k = size(X2)
Q = ones(p, 1)
    for i in 2:k
        if i in ind
            i = i+1
        else
            Q = hcat(Q, X2[:, i])
        end
    end
    return(Q)
end

X2_REDUCT = myreduction(X_2, ind)

Ttest = X2_REDUCT*transpose(randmat)
Ttest[Ttest .< 0] .= 0
XTest_new = [X2_REDUCT Ttest]

predict_matrix = XTest_new*beta
r, c = size(predict_matrix)  
my_predict = zeros(r,1)    


for i in 1:r
        u1 = argmax(predict_matrix[i, :])
        my_predict[i] = u1-1
end

 my_predict

test_y

sum(my_predict.==test_y)/10000
