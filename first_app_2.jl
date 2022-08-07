using Convex, SCS, MLDatasets, LinearAlgebra

using InvertedIndices

train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

a1 = permutedims(train_x, [2,1,3])
a2 = transpose(reshape(a1, (784, 60000)))
X=hcat(ones(60000),a2)


Nval = 4000;
X_val = X[1:Nval,:] 
Y_val = train_y[1:Nval]
X_tr = X[20001:60000,:] 
Y_tr = train_y[20001:60000]
q,m=size(X_tr);


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

randmat = [rand() < 0.5 for _ in 1:1500, _ in 1:fe]
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

n_iter=10

#row_val = rand(1:60000, 1000)


#row_tr = Not(rwo_val)

function Ytrain(i, train_y)

    d=-1*ones(length(train_y))
    for j in 1:length(train_y)
        if train_y[j] == i
        d[j] = 1
        end        
    end
    B=vcat(d,zeros(m))
    return(B)
end


  #A_tr=vcat(X_tr ,exp(w)*I(m))
  #A_val=X_val


function theta_ls(w, B)
    theta=Variable(m)
    objective =norm(vcat(X_tr,exp(w)*I(m))*theta-B)
    problem = minimize(objective)
    solve!(problem, SCS.Optimizer,silent_solver = true)
    return(Matrix(theta.value))
end

function theta_ls2(w, B)
   A = vcat(X_tr ,exp(w)*I(m));
   return(A\B)
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
    return(inv(L)*(DAT(X_tr,w)-DL(X_tr, w)*(inv(L)*transpose(A)))*B)

end

function g(theta,w,B)
    A = Matrix(vcat(X_tr ,exp(w)*I(m)));
    Dsi = -1/Nval*transpose(2*(Y_val-Yh_val(theta)))*(X_val*Dtheta(A,B, w))
    return(Dsi)
end

function F(Y_val, theta)
    si = 1/Nval*(norm(Y_val-Yh_val(theta))^2)
    return(si)
end

function iterr(n_iter, w, B)
    t=1;
    eps=.01;
    for i in 1:n_iter
        temp=theta_ls2(w, B)
        w_tent=w-t*g(temp,w,B)
        if F(Y_val,theta_ls2(w_tent, B))<=F(Y_val,temp)
            t=1.2*t
            temp2=w
            w=w_tent
            if norm((temp2-w)/t+g(temp,w,B)-g(temp,temp2,B))<=eps
                break
            end
        else
            t=t/2
        end
    end
    return(w)
end

beta = zeros(m, 10);

v,m = size(X_tr)
for i in 0:1:9
    w=-2;
    A = Matrix(vcat(X_tr ,exp(w)*I(m)));
    B = Ytrain(i, Y_tr);
    w = iterr(n_iter, w, B);
    beta[:,i+1]=theta_ls2(w, B);

end

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
X2new = [X2_REDUCT Ttest]

predict_matrix = X2new*beta
r, c = size(predict_matrix)  
my_predict = zeros(r,1)    


for i in 1:r
        u1 = argmax(predict_matrix[i, :])
        my_predict[i] = u1-1
end

 my_predict

test_y

sum(my_predict.==test_y)/10000
