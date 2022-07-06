import torch
import torch.optim as optim

a = [1,2,3,4]
b=[4]
print(set(a)-set(b))
testvar=torch.ones([])
testvar.requires_grad=True

testvar_copy=testvar.clone().detach().requires_grad_(True)
print(testvar_copy)
for i in range(5):
    print(i)
    opt=optim.SGD([testvar_copy],1e-3)
    print(testvar_copy)

    testvar_copy.backward()
    opt.step()
    print(testvar_copy)
    testvar_copy=testvar.clone().detach().requires_grad_(True)
