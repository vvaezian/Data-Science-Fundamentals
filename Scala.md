```scala
print("Hello World")  // use double-quotes for strings
                      // use 'println` to inser a new line at the end
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
### Arrays (mutable, like lists in Python)
```scala
val a: Array[Int] = new Array[Int](2)  // Array(0, 0) 
                                       // use the type 'Any' to be able to use elements of mixed types in the array
a(0) = 5  # parathesis is used instead of square brackets as is Java and Python
a(1) = 6
```
Shorthand: `val b = Array(12, 10, 8)`  
While the array elements can change, to make the array itself immutable we should define it with `val`.

```scala
var b = Array(0, 1)  // we use var to be able to reassign values to b, but Scala suggests using val and creating new variables instead of re-assigning.
b.foreach(println)  // print values of the array
b = b :+ 3  // appending an element
b = 2 +: b // prepending an element
b = b ++ Array(1, 2) 
```

### Lists (immutable)