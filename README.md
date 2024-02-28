# hx
```
#全连接层
class Connect(nn.Module):
    def __init__(self, input_c: int):
        super(Connect, self).__init__()
        self.op1 = nn.Conv2d(input_c,31,5,1,3)
        self.bn1 = nn.BatchNorm2d(31)
        self.op2 = nn.Conv2d(31,9,1,1,0)
        self.bn2 = nn.BatchNorm2d(9)
        self.op3= nn.Conv2d(9,3,1,1,0)
        self.bn3 = nn.BatchNorm2d(3)
        self.op4= nn.Conv2d(3,1,1,1,0)
        self.bn4 = nn.BatchNorm2d(1)
        self.op5 = nn.Linear(5625,2592)
        self.bn5 = nn.BatchNorm1d(2592)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.op1(x)
        # x = self.relu(x)
        x = self.op2(x)
        # x = self.relu(x)
        x = self.op3(x)
        # x = self.relu(x)
        x = self.op4(x)
        # x = self.relu(x)
        x = self.flat(x)
        x = self.op5(x)
        # x = self.bn5(x)
        x = torch.reshape(x,((-1,2,36,36)))
        return x

class Connect2(nn.Module):
    def __init__(self):
        super(Connect2, self).__init__()
        self.op1 = nn.Conv2d(2,1,1,1,0)
        self.bn1 = nn.BatchNorm2d(1)
        self.op2 = nn.Linear(1296,2000)
        self.bn2 = nn.BatchNorm1d(2000)
        self.op3 = nn.Linear(2000,548)
        self.bn3 = nn.BatchNorm1d(548)
        self.op4 = nn.Linear(548,1367)
        self.bn4 = nn.BatchNorm1d(1367)
        self.op5 = nn.Linear(1367,2592)
        self.bn5 = nn.BatchNorm1d(2592)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.op1(x)
        # x = self.relu(x)
        x = self.flat(x)
        x = self.op2(x)
        # x = self.bn2(x)
        x = self.op3(x)
        # x = self.bn3(x)
        x = self.op4(x)
        # x = self.bn4(x)
        x = self.op5(x)
        # x = self.bn5(x)
        x = torch.reshape(x,((-1,2,36,36)))   
        return x
class Connect3(nn.Module):
    def __init__(self, input_c: int):
        super(Connect3, self).__init__()
        self.op1 = nn.Conv2d(4,27,1,1,3)
        self.bn1 = nn.BatchNorm2d(27)
        self.op2 = nn.Conv2d(27,20,5,1,2)
        self.bn2 = nn.BatchNorm2d(20)
        self.op3 = nn.Conv2d(20,4,3,1,2)
        self.bn3 = nn.BatchNorm2d(4)
        self.op4 = nn.Conv2d(4,1,1,1)
        self.bn4 = nn.BatchNorm2d(1)
        self.op5 = nn.Linear(1936,1144)
        self.bn5 = nn.BatchNorm1d(1144)
        self.op6 = nn.Linear(1144,2592)
        self.bn6 = nn.BatchNorm1d(2592)
        self.op7 = nn.Conv2d(2,30,3,2,0)
        self.bn7 = nn.BatchNorm2d(30)
        self.op8 = nn.Conv2d(30,8,1,1,0)
        self.bn8 = nn.BatchNorm2d(8)
        self.op9 = nn.Linear(2312,1323)
        self.bn9 = nn.BatchNorm1d(1323)
        self.op10 = nn.Linear(1323,10658)
        self.bn10 = nn.BatchNorm1d(10658)
     
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.op1(x)
        # x = self.relu(x)
        x = self.op2(x)
        # x = self.relu(x)
        x = self.op3(x)
        # x = self.relu(x)
        x = self.op4(x)
        # x = self.relu(x) 
        x = self.flat(x)

        x = self.op5(x)
        # x = self.bn5(x)
        x = self.op6(x)
        # x = self.bn6(x)
        x = torch.reshape(x,((-1,2,36,36))) 
        x = self.op7(x)
        # x = self.relu(x)
        x = self.op8(x)
        # x = self.relu(x)
        x = self.flat(x)
        x = self.op9(x)
        # x = self.bn9(x)
        x = self.op10(x)
        # x = self.bn10(x)
        x = torch.reshape(x,((-1,2,73,73))) 
        return x
class CNT(nn.Module):
    def __init__(self,input_c: int):
        super(CNT,self).__init__()
        self.a = Connect(input_c)
        self.b = Connect2()
        self.c = Connect3(input_c)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.a(x)
        x2 = self.b(x1)
        x3 = torch.cat([x1,x2],dim=1)
        x4 = self.c(x3)
        return x4
'''
