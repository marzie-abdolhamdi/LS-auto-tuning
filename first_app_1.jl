using MLDatasets, LinearAlgebra, Convex, SCS

train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

a1 = permutedims(train_x, [2,1,3])
a2 = transpose(reshape(a1, (784, 60000)))
X=hcat(ones(60000),a2)


q,m=size(X);

ind = [1]
L = ones(q, 1)

for i in 2:m
    v = count(x -> x>0, X[:,i])
    if v<600
        ind = hcat(ind,[i])
    else
        L = hcat(L, X[:,i])
    end
            
end
ind = ind[2:end]

randmat = [rand() < 0.5 for _ in 1:1000, _ in 1:494]
randmat = round.(Int, randmat)
randmat[randmat .== 0] .= -1

T = L*transpose(randmat)
T[T .< 0] .= 0
Xnew = [L T]

btest1 = permutedims(test_x, [2,1,3])
btest2 = transpose(reshape(btest1, (784, 10000)))
X_2= hcat(ones(10000),btest2)

n_iter=5;
q, m = size(Xnew)

beta = zeros(m, 10)

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


@time theta_ls(-2, B)

@time theta_ls2(-2, B)

function theta_ls(w, B)
    theta=Variable(m)
    objective =norm(vcat(Xnew,exp(w)*I(m))*theta-B)
    problem = minimize(objective)
    solve!(problem, SCS.Optimizer,silent_solver = true)
    return(Matrix(theta.value))
end

function theta_ls2(w, B)
   A=vcat(Xnew,exp(w)*I(m))
   return(A\B)
end

function g(theta,w)
    return(exp(w)*norm(theta))
end

function F(theta,w, B)
    Z=vcat(Xnew,exp(w)*I(m))
    return(norm(Z*theta-B))
end

function iterr(n_iter, w, B)
    t=1;
    eps=.01;
    for i in 1:n_iter
        temp=theta_ls2(w, B)
        w_tent=w-t*g(temp,w)
        if F(temp,w_tent, B)<=F(temp,w, B)
            t=1.2*t
            temp2=w
            w=w_tent
            if norm((temp2-w)/t+g(temp,w)-g(temp,temp2))<=eps
                break
            end
        else
            t=t/2
        end
    end
    return(w)
end

for i in 0:1:9
    w=-2;
    B = Ytrain(i, train_y);
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

sum(my_predict.==test_y)/10000
