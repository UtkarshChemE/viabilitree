# Viabilitree

This library propose a set of algorithms, based on a particular kd-tree structure, in order to compute viability kernels and capture basins.

## Motivation
Mathematical viability theory offers concepts and methods that are suitable to study the compatibility between a dynamical system described by a set of differential equations and constraints in the state space. The result sets built during the viability analysis can give very useful information regarding management issues in fields where it is easier to discuss constraints than objective functions. Viabilitree is a framework in which the viability sets are represented and approximated with particular kd-trees. The computation of the viability kernel is seen as an active learning problem. We prove the convergence of the algorithm and assess the approximation it produces for known problems with analytical solution. This framework aims at simplifying the declaration of the viability problem and provides useful methods to assist further use of viability sets produced by the computation.


## Viability problem

### Simple example
#### Population Growth Model
This example is taken from [4]. The population model is Malthusian. The population viability problem consists in maintaining the size of a population in a given interval <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;[a;b]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;[a;b]" title="[a;b]" /></a>. The state of the system is described by the variables <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x(t)" title="x(t)" /></a>, the size of the population, and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;y(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;y(t)" title="y(t)" /></a>, the population growth rate. The dynamics are described by the following equations:

<a href="https://www.codecogs.com/eqnedit.php?latex=\left\{&space;\begin{array}{lll}&space;x(t&plus;dt)&space;&=&&space;x(t)&plus;x(t)y(t)dt\\&space;y(t&plus;dt)&space;&=&&space;y(t)&plus;u(t)dt&space;\text{&space;with&space;}&space;\left|&space;u(t)&space;\right|&space;\leq&space;c&space;\end{array}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left\{&space;\begin{array}{lll}&space;x(t&plus;dt)&space;&=&&space;x(t)&plus;x(t)y(t)dt\\&space;y(t&plus;dt)&space;&=&&space;y(t)&plus;u(t)dt&space;\text{&space;with&space;}&space;\left|&space;u(t)&space;\right|&space;\leq&space;c&space;\end{array}\right." title="\left\{ \begin{array}{lll} x(t+dt) &=& x(t)+x(t)y(t)dt\\ y(t+dt) &=& y(t)+u(t)dt \text{ with } \left| u(t) \right| \leq c \end{array}\right." /></a>

The dynamics are controlled by taking the growth rate evolution in interval <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;[-c,c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;[-c,c]" title="[-c;c]" /></a>. This viability problem can be resolved analytically (see [4]} for details). When $`dt`$ tends toward $`0`$, the theoretical viability kernel is defined by:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{Viab}(K)&space;=&space;\left\{&space;(x,y)\in&space;{\mathbb&space;R}^2|&space;\quad&space;x&space;\in&space;[a;b],&space;y\in&space;\left[-\sqrt{2c\log\frac{x}{a}};&space;\sqrt{2c\log\frac{b}{x}}\right]&space;\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{Viab}(K)&space;=&space;\left\{&space;(x,y)\in&space;{\mathbb&space;R}^2|&space;\quad&space;x&space;\in&space;[a;b],&space;y\in&space;\left[-\sqrt{2c\log\frac{x}{a}};&space;\sqrt{2c\log\frac{b}{x}}\right]&space;\right\}" title="\text{Viab}(K) = \left\{ (x,y)\in {\mathbb R}^2| \quad x \in [a;b], y\in \left[-\sqrt{2c\log\frac{x}{a}}; \sqrt{2c\log\frac{b}{x}}\right] \right\}" /></a>


<img src="images/populationGitlab.png" width="300" alt="Figure 1: Viability kernel of the population viability problem">[Figure 1: Viability kernel of the population viability problem][Figure 1]

[//]: # (![Approximation of the population growth viability kernel](images/populationGitlab.png "Direct approximation")*Figure 1: Viability kernel of the population viability problem*)


The above figure shows an approximation of the viability kernel for the population problem with:
* constraint set <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;K=[a=0.2,b=3]\times[d=-2,e=2]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;K=[a=0.2,b=3]\times[d=-2,e=2]" title="K=[a=0.2,b=3]\times[d=-2,e=2]" /></a>, 
* parameters <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;dt=0.1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;dt=0.1" title="dt=0.1" /></a>, 
* control set <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;U=[-0.5;0.5]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;U=[-0.5;0.5]" title="U=[-0.5;0.5]" /></a> with discretization step 0.02. 
The color stands for the value of a control <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;u" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;u" title="u" /></a> which allows the state to stay in the viability kernel. In black the boundary of the true kernel.

**By definition of the viability kernel, starting from any point in the viability kernel, there exists always an evolution that stays in the viability kernel. Starting from any point outside the viability kernel, any evolution will exit the constraint set in finite time.** This is why the notion of viability kernel is so useful.

The corresponding code is the following:

For the definition of the model: dynamics, perturbations, etc.
```scala
import viabilitree.model.Dynamic

case class Population(integrationStep: Double = 0.01, timeStep: Double = 0.1) {

  def dynamic(state: Vector[Double], control: Vector[Double]) = {
    def xDot(state: Vector[Double], t: Double) = state(1) * state(0)
    def yDot(state: Vector[Double], t: Double) = control(0)
    val dynamic = Dynamic(xDot, yDot)
    val res = dynamic.integrate(state.toArray, integrationStep, timeStep)
    res
  }
```
_timeStep_ stands for to $dt$.
_integrationStep_ is a private parameter used by the _integrate_ method.


For the definition of the viability problem:
```scala
import scala.util.Random
import viabilitree.viability._
import viabilitree.export._
import viabilitree.kdtree.Tree
import viabilitree.viability.kernel._
import java.io.File

object PopulationViability extends App {
  // accuracy parameter
  val depth = 20
  
  // algorithm parameter  
  val rng = new Random(42)
  
  // model definition  
  val population = Population()
  
  // control parameter  
  def c = 0.5
  
  // constraint set parameters 
  def a = 0.2
  def b = 3.0
  def d = -2.0
  def e = 2.0
  
  // definition of the viability problem
  val vk = KernelComputation(
    dynamic = population.dynamic,
    depth = depth,
    zone = Vector((a, b), (d, e)),
    controls = Vector(-c to c by 0.02)
  )

  // computation of the viability kernel corresponding to problem vk
  val (ak, steps) = approximate(vk, rng)
  
  // save viability kernel to file (vtk format, to be processed by paraview)
  val f = new File(s"population${steps}depth${depth}.vtk")
  saveVTK2D(ak, f)
}
```
_a_ to _e_ are the same parameters as in the mathematical definition.

The viability problem is defined by class _KernelComputation_ with the following parameters:

* _depth_ which defines the accuracy of the approximation. There are <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;2^{depth}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;2^{depth}" title="2^{depth}" /></a> grid points (here, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;2^{\frac{depth}{2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;2^{\frac{depth}{2}}" title="2^{\frac{depth}{2}}" /></a> points per axes).
* _dynamic_: the model dynamic
* _zone_: the area to explore and here it is also the constraint set, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;[a,b]\times[d,e]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;[a,b]\times[d,e]" title="[a,b]\times[d,e]" /></a>
* _controls_: the set of admissible controls, it is the same set for each state,<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;[-c,c]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;[-c,c]" title="[-c,c]" /></a>

The computation itself is done by the _approximate_ function.

### Mathematical Viability Theory ([2], [3])
<a id="MVT"></a>
In Viabilitree we consider a viability problem defined by a controlled dynamical system <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;S" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;S" title="S"/></a>, a set-valued map <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;U" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;U" title="U"/></a> (the set of admissible controls depending on the state of the system), and a compact subset <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;K" title="K"/></a> of the state space (the set of constraints):

<a href="https://www.codecogs.com/eqnedit.php?latex=(S)\left\{&space;\begin{array}{lll}&space;x'(t)&=&\Phi(x(t),u(t))\\&space;u(t)&\in&space;&&space;U(x(t))&space;\end{array}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?(S)\left\{&space;\begin{array}{lll}&space;x'(t)&=&\Phi(x(t),u(t))\\&space;u(t)&\in&space;&&space;U(x(t))&space;\end{array}\right." title="(S)\left\{ \begin{array}{lll} x'(t)&=&\Phi(x(t),u(t))\\ u(t)&\in & U(x(t)) \end{array}\right." /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x(t)" title="x(t)"/></a> is the state of the system <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;S" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;S" title="S"/></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x(t)\in&space;{\mathbb&space;R}^p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x(t)\in&space;{\mathbb&space;R}^p" title="x(t)\in {\mathbb R}^p" /></a> a finite dimensional vector space.
 <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;u(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;u(t)" title="u(t)" /></a> is the control, with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;u(t)\in&space;{\mathbb{R}}^q" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;u(t)\in&space;{\mathbb{R}}^q" title="u(t)\in {\mathbb{R}}^q" /></a>.
 The set-valued map <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;U&space;:&space;{\mathbb&space;R}^p\leadsto&space;{\mathbb{R}}^q" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;U&space;:&space;{\mathbb&space;R}^p\leadsto&space;{\mathbb{R}}^q" title="U : {\mathbb R}^p\leadsto {\mathbb{R}}^q" /></a> gives the set of admissible control for each state <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x" title="x" /></a>. <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\Phi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\Phi" title="\Phi" /></a> is a function from <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\text{Graph}(U)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\text{Graph}(U)" title="\text{Graph}(U)" /></a> to <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\mathbb&space;R}^p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;{\mathbb&space;R}^p" title="{\mathbb R}^p" /></a>.
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;K\subset&space;{\mathbb&space;R}^p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;K\subset&space;{\mathbb&space;R}^p" title="K\subset {\mathbb R}^p" /></a> is a compact subset of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\mathbb&space;R}^p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;{\mathbb&space;R}^p" title="{\mathbb R}^p" /></a>, it is the set of desirable states, the constraint set in which the state <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x(t)" title="x(t)" /></a> is supposed to stay.

The viability kernel <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\text{viab}_S(K)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\text{viab}_S(K)" title="\text{viab}_S(K)" /></a> is the largest subset of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;K" title="K" /></a> (possibly empty) that gathers the states from which it is possible to find a control function <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;u(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;u(t)" title="u(t)" /></a> such that the evolution <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x(.)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x(.)" title="x(.)" /></a> stays in the compact set <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;K" title="K" /></a>.

<a href="https://www.codecogs.com/eqnedit.php?latex=x\in&space;\text{viab}_S(K)&space;\Leftrightarrow&space;\exists&space;u(.)&space;\quad&space;\forall&space;t\geq&space;0&space;\left\{&space;\begin{array}{lll}&space;x'(t)&=&\Phi(x(t),u(t))\\&space;u(t)&\in&space;&&space;U(x(t))\\&space;x(t)&\in&space;&&space;K&space;\end{array}\right." target="_blank"><img src="https://latex.codecogs.com/gif.latex?x\in&space;\text{viab}_S(K)&space;\Leftrightarrow&space;\exists&space;u(.)&space;\quad&space;\forall&space;t\geq&space;0&space;\left\{&space;\begin{array}{lll}&space;x'(t)&=&\Phi(x(t),u(t))\\&space;u(t)&\in&space;&&space;U(x(t))\\&space;x(t)&\in&space;&&space;K&space;\end{array}\right." title="x\in \text{viab}_S(K) \Leftrightarrow \exists u(.) \quad \forall t\geq 0 \left\{ \begin{array}{lll} x'(t)&=&\Phi(x(t),u(t))\\ u(t)&\in & U(x(t))\\ x(t)&\in & K \end{array}\right." /></a>

In Viabilitree we follow the method described in [5], we consider dynamical system (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;S_{dt}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;S_{dt}" title="S_{dt}" /></a>) discretized in time: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{array}{lll}&space;F_{dt}(x)&=&&space;\left\{&space;x&plus;\Phi(x,u)dt,&space;u&space;\in&space;U(x)&space;\right\}.&space;\end{array}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{array}{lll}&space;F_{dt}(x)&=&&space;\left\{&space;x&plus;\Phi(x,u)dt,&space;u&space;\in&space;U(x)&space;\right\}.&space;\end{array}" title="\begin{array}{lll} F_{dt}(x)&=& \left\{ x+\Phi(x,u)dt, u \in U(x) \right\}. \end{array}" /></a>

We use the learning algorithm <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;L" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;L" title="L" /></a> of kd-tree described in [1] on a discretized grid <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;K_h" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;K_h" title="K_h" /></a> to compute an approximation <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;L(K_h)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;L(K_h)" title="L(K_h)" /></a> of the viability kernel <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\text{viab}_{S_{dt}}(K)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\text{viab}_{S_{dt}}(K)" title="\text{viab}_{S_{dt}}(K)" /></a> of the discretized dynamical system (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;S_{dt}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;S_{dt}" title="S_{dt}" /></a>) with constraint set <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;K" title="K" /></a>. When the learning algorithm and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;S_{dt}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;S_{dt}" title="S_{dt}" /></a> verify some conditions, [1], [5] and [6] ensure that <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;L(K_h)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;L(K_h)" title="L(K_h)" /></a> converges to <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;viab_S(K)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;viab_S(K)" title="viab_S(K)" /></a> when <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;h" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;h" title="h" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;dt" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;dt" title="dt" /></a> tend to 0.

The convergence conditions are:
 * <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;F" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;F" title="F" /></a> is [Marchaud][Marchaud] and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\mu" title="\mu" /></a>-[Lipschitz][Lipschitz] with closed images
 * <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;K" title="K" /></a> is a compact set.
 * <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;L(K^0)=K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;L(K^0)=K" title="L(K^0)=K" /></a>. 
 * <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;viab_S(K)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;viab_S(K)" title="viab_S(K)" /></a> is compact, it is path-connected 
 * <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;viab_S(K)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;viab_S(K)" title="viab_S(K)" /></a> erosion with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;B(\epsilon)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;B(\epsilon)" title="B(\epsilon)" /></a> is path-connected and points of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;viab_S(K)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;viab_S(K)" title="viab_S(K)" /></a> are at most distant from the eroded set by <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\epsilon&space;\sqrt{p}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon&space;\sqrt{p}" title="\epsilon \sqrt{p}" /></a>
 * <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;viab_S(K)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;viab_S(K)" title="viab_S(K)" /></a> complementary set is path-connected as is its erosion with <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;B(\epsilon)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;B(\epsilon)" title="B(\epsilon)" /></a>, and its points are at most distant from the eroded set by <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\epsilon&space;\sqrt{p}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon&space;\sqrt{p}" title="\epsilon \sqrt{p}" /></a>.
The two last properties ensure that there are no small tentacles.

In practice points of the grid are removed from the current dilated approximation when following the dynamic with each available control they always leave the dilated approximation at the next step. See [KernelComputation][kernelLink] for implementation details.

More details on both theoretical and practical aspects in [7][viabilitree]

## Install
Coming soon
<!-- A rédiger après la mise en forme finale -->

## Use
The main purpose of **Viabilitree** is to help users to declare a viability problem, compute the corresponding viability kernel and possibly its capture basin, in order to perform a viability analysis.

Example of viability problems are gathered in package **example**. Each example is detailled in its own readme file.
 * [Bilingual][Bilingual] :
   - approximation of the viability kernel (BilingualViabDomain)
   - approximation of the capture basin of the viability kernel (BilingualBasin)
 * [Consumer][Consumer]
   - approximation of the viability kernel (ConsumerViability)
   - approximation of a set (the analytical kernel ) (ConsumerKernel)
 * [Lake][Lake]
   - approximation of the viability kernel (LakeViability)
   - use of output files (OutputLake)
 * [Population][Population]
   - approximation of the viability kernel (PopulationViability)
   - exploration with OpenMOLE (PopulationViability)
   - approximation of a set (the analytical kernel ) (PopulationApproximation)

## Set operators

 * volume
 * intersection of trees
 * indicator function
 * erosion
 * dilation

Examples in [circle][circle].
 
## Learning algorithm
Apart from viability study it is possible to use **Viabilitree** as a simple learning algorithm.

See [approximation][approximation] package for more details.

Examples in [population] and [circle]

To be completed

<!-- A rédiger après la mise en forme finale -->

#### References
[1] Rouquier et al, A kd-tree algorithm to discover the boundary of a black box hypervolume, _Annals of Mathematics and Artificial Intelligence_, vol 75, _3_, pp "335--350, 2015. 

[2] Aubin, _Viability theory_, Birkhäuser, 1991

[3] Aubin, Bayen, A. and Saint-Pierre, P. _Viability Theory: New Directions_, Springer, 2011.

[4] Aubin et Saint-Pierre, "An introduction to viability theory and management of renewable resources", chapter in _Advanced Methods for Decision Making and Risk Management_, J. Kropp and J. Scheffran eds., Nova Science Publishers, 2007

[5] Deffuant , Chapel,  Martin (2007). Approximating viability kernels with support vector machines. _IEEE T. Automat. Contr._, 52(5), 933–937.

[6] Saint-Pierre, P. (1994). Approximation of the viability kernel. _Applied Mathematics & Optimisation_, 29(2), 187–209.

[7] Alvarez, Reuillon, de Aldama. Viabilitree: A kd-tree Framework for Viability-based Decision. hal-01319738.

#### Remarks
<a name="Marchaud"></a>
A set-valued map $`F`$, non trivial, upper semicontinuous, with compact convex images is a Marchaud map if it has linear growth, that is there exists $`c > 0`$ such that $`\forall x, \  \sup _{y \in F(x)} \left\|y\right\| \leq c(\left\|x\right\|+1)`$.

<a name="Lipschitz"></a>
A set-valued map $`F`$ is $`\mu`$-Lipschitz with $`\mu>0`$ if for all $`x`$ and $`y`$, $` F(x)\subset F(y)+B(0,\mu ||x-y||)`$

<!-- Identifiers, in alphabetical order -->
[approximation]: doc/READMEApproximation.md "doc for approximation package"
[Bilingual]:https://gitlab.iscpif.fr/viability/viabilitree/tree/master/example/bilingual/src/main/scala/viabilitree/example/bilingual "Documentation for the example of the Bilingual society viability problem"
[circle]:https://gitlab.iscpif.fr/viability/viabilitree/tree/master/example/circle/src/main/scala/viabilitree/approximation/example/circle "Documentation for the example of approximation problems (includning intersection)"
[Consumer]:example/consumer/READMEconsumer.md "Documentation for the example of the consumption model"
[kernelLink]:doc/READMEKernelComputation.md "Documentation for class KernelComputation"
[Lake]:https://gitlab.iscpif.fr/viability/viabilitree/tree/master/example/lake/src/main/scala/viabilitree/approximation/example/lake/READMElake.md "Eutrophication and lakeside farms problem"
[Lipschitz]: #Lipschitz "Definition of the Lipschitz property for dynamical systems"
[Marchaud]: #Marchaud "Definition of the Marchaud property for dynamical systems"
[Population]:https://gitlab.iscpif.fr/viability/viabilitree/tree/master/example/population/src/main/scala/fr/iscpif/population/READMEpopulation.md "Documentation for the example of the population growth model"
[viabilitree]: https://hal.archives-ouvertes.fr/hal-01319738v1 "Working paper with technical proofs"
<!-- [openmole]: http://www.openmole.org/ "OpenMOLE website: for numerical exploration of complex models" -->
