 <center>

# Luna Regression Tutorial

</center>

Based on https://www.tensorflow.org/tutorials/keras/basic_regression

Dataset was downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data and it has been slightly modified to be loaded out of the box.

## Cloning repository.

```bash
git clone https://github.com/Luna-Tensorflow/RegressionTutorial.git
git clone -b MNIST_tutorial https://github.com/Luna-Tensorflow/Luna-Tensorflow.git
cd RegressionTutorial
```

## Building libraries.
```bash
cd local_libs/Tensorflow/native_libs/
mkdir build
cd build
cmake ../src
make
cd ../../../..
```

```
import Std.Base
import Dataframes.Table
import Dataframes.Column
import Tensorflow.Layers.Input
import Tensorflow.Layers.Dense
import Tensorflow.Optimizers.RMSProp
import Tensorflow.Losses.MeanError
import Tensorflow.Model
import Tensorflow.Tensor
import Tensorflow.Types
import Tensorflow.Operations
import Tensorflow.GeneratedOps
import RegressionTutorial.DblColumn

def extendWith table name value:
    table' = table.eachTo name (row: (row.at "Origin" == value).switch 0.0 1.0)
    table'

def oneHotOrigin table:
    t1 = extendWith table "USA" 1
    t2 = extendWith t1 "Europe" 2
    t3 = extendWith t2 "Japan" 3
    t3

def shuffle table:
    row = table.rowCount
    rand = Tensors.random FloatType [row] 0.0 0.0
    col = columnFromList "rand" (rand.toFlatList)
    table1 = table.setAt "rand" col
    table2 = table1.sort "rand"
    table3 = table2.remove "rand"
    table3

def sample table fracTest:
    testCount = (fracTest * table.rowCount.toReal).floor
    test = table.take testCount
    train = table.drop testCount
    (train, test)

def nfeatures:
    9

def convertToTf shape table:
    "this is a workaround until we get native Dataframes <-> TF integration"
    lst = table.toList . each (col: col.toList)
    t1 = Tensors.fromList2d FloatType lst
    t2 = Tensors.transpose t1
    lst' = Tensors.to2dList t2
    samples = lst'.each(l: Tensors.fromList FloatType shape l)
    samples

def main:
    print "Loading data"
    
    table = Table.read "auto-mpg.csv"
    table1 = table.dropNa
    table2 = oneHotOrigin table1
    table3 = table2.remove "Origin"
    table4 = shuffle table3
    (trainTable, testTable) = sample table4 0.2
    print "TODO: normalize both tables using stats from train"

    trainLabels = trainTable.at "MPG"
    testLabels = testTable.at "MPG"
    trainTable' = trainTable.remove "MPG"
    testTable' = testTable.remove "MPG"

    trainX = convertToTf [nfeatures, 1] trainTable'
    testX = convertToTf [nfeatures, 1] testTable'
    trainY = convertToTf [1, 1] trainLabels
    testY = convertToTf [1, 1] testLabels

    print "Building net"
    i = Input.create FloatType [nfeatures, 1]
    d1 = Dense.createWithActivation 64 Operations.relu i
    d2 = Dense.createWithActivation 64 Operations.relu d1
    d3 = Dense.createWithActivation 1 Operations.relu d2

    lr = 0.001
    rho = 0.9
    momentum = 0.0
    epsilon = 0.000000001
    opt = RMSPropOptimizer.create lr rho momentum epsilon

    loss = MeanErrors.meanSquareError

    model = Models.make i d3 opt loss


    print "Training"
    epochsN = 10
    epochs = 1.upto epochsN
    fitted = epochs.foldLeft model (epoch: model: model.train trainX trainY)

    print "Evaluation"
    predY = testX . each (tx: fitted.evaluate tx . head . get)

    print (testY.toJSON)
    errorSum = (predY . zip testY) . foldLeft 0.0 ((pY, tY): ((absPatch (pY.atIndex 0 - tY.atIndex 0)) +) )
    maxn = testX.length . toReal
    meanErr = errorSum / maxn
    print ("Mean error: " + (meanErr.toText))
    None

def absPatch x:
  if x < 0.0 then x.negate else x

```

![](Screenshots/main.png)