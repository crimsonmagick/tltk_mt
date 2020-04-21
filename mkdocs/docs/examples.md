# Examples

## MTL Formulas 

The TLTk uses python objects to represent MTL operations. The currently supported MTL operarations are:

### Predicate Definition:

``` python
MTL.Predicate(dataName, AMatrix, bMatrix) 
```
The predicate is defined as a polyhedron in the form Ax ≤ b. For higher diminsional problems dataName can be a list of names and TLTk will zip those rows together. It is recommended to have the data in the correct format before hand and not zip for speed.

For example, we can define 

``` python
speedPred = Predicate('speed', AMatrix, bMatrix) 
```

``` python
rpmPred = Predicate('rpm', AMatrix, bMatrix) 
```

### MTL Operators

Not:

``` python
phi = MTL.Not(speedPred)
```

And: 

``` python
phi = MTL.And(speedPred,rpmPred)
```

Or:

``` python
phi = MTL.Or(speedPred,rpmPred)
```

Next:

``` python
phi = MTL.Next(speedPred)
```

Eventually:

``` python
phi = MTL.Finally(lowerTimeBound,upperTimeBound,Subformula) 
```
The first argument is the lower time bound. The second argument is the upper time bound and can be float(’inf’) to represent infinity. The third argument is either a predicate or another MTL operation.

Always:

``` python
phi = MTL.Globaly(lowerTimeBound,upperTimeBound,Subformula) 
```
The first argument is the lower time bound. The second argument is the upper time bound and can be float(’inf’) to represent infinity. The third argument is either a predicate or another MTL formula.

Until:

``` python
phi = MTL.Until(lowerTimeBound,upperTimeBound,Subformula1,Subformula2) 
```
The first argument is the lower time bound. The second argument is the upper time bound and can be float(’inf’) to represent infinity. The third argument is either a predicate or another MTL operation. The fourth argument is either a predicate or another MTL formula.

## Simple MTL Formula Example

We define a simple predicate as follows:

``` python
r1 = MTL.Predicate('data1', [1, 1], [150]) 
```
Mathematically, this defines the set: 

$$r1:
\begin{bmatrix}
1 & 1
\end{bmatrix}
\begin{bmatrix}
x_1 \\ 
x_2
\end{bmatrix}
\le 150
$$

Where $x_1$ and $x_2$ are the signals the trajectory is defined over. 

We define the MTL formula $\varphi=F_{[0,3.14]} (r1)$ as: 

``` python
phi = MTL.Eventually(0,3.14,r1) 
```

## Automotive Transmission

Bla Bla

## Redex


