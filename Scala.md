- Scala combines object-oriented and functional programming. Its style is a hybrid of functional and imperative programming, with the functioanl style preferred.
- Scala code is first compiled to Java byte codes, then run on a JVM

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
In Scala, if there is no `return` then the last expression is taken to be the return value. In the example above the `if-else` is the last expression.  
If we include the `return`, specifying the return type of the function becomes necessary.

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
b.foreach(println)  // print values of the array. This can be done with a loop as well (imperative).
b = b :+ 3  // appending
b = 2 +: b // prepending
b = b ++ Array(1, 2) 
b.length  // length of array
```

### Lists (immutable)
Lists are linked lists. The prepend operation is O(1) as expected but append operation is O(n). This is because in Scala list elements are immutable, and to "modify" the last element of the list (changing the pointer), we need to make a copy of it first.  
This is why prepend is more common for lists. If we need to append we should use other data types like ListBuffer.
```scala
val a = List(1, 2)
0 :: a  // prepending (:: is called the 'cons' operator)
a :+ 3  // appending. This is not efficient (O(n)). Use ListBuffer or other alternatives
List(1,2) ::: List(3, 4)  // concatenate
```
`Nil` is an empty list.  
Another way to initialize lists is using the cons operator and Nill:
```scala
val c = 1 :: 2 :: Nil
```
### `if`
```scala
if (a > 10) println(a)  // if there is only one statement, we can write it in one line
                        // but it is recommended to use curly braces when there is no "else" statement
if (a > 10)
  println(a)
else
  println("No")
if (a > 10) {  // if the are more than one statement we have to to use curly braces
  println(a)
  println("yes")
}
else
  println("No")

val max = if (b > c) b else c  // this can be more elaborate as well (e.x. if-else if-else)
```

