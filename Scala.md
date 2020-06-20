```scala
print("Hello World")  // use double-quotes for strings
```
There are two types of variables: `var` (mutable), `val` (immutable):
```scala
val my_immutable_variable_name: Int = 10  // shorter version: val my_immutable_variable_name = 10
var my_mutable_variable_name: Int = 20  // shorted version: var my_mutable_variable_name = 20
```

Common Types for data-related tasks: `Double`, `Int`, `Boolean`, `String`

### Functions
```scala
def myFunc(param1: Int, param2: Int): Int = {  // the ": Int " part is not necessary
  if (param1 > param2) param1
  else param2
}
```
