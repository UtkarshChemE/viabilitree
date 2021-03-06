package fr.iscpif.population

/*
import fr.iscpif.viability._
import fr.iscpif.kdtree.algorithm._
import fr.iscpif.kdtree.structure._
import fr.iscpif.kdtree.content._
*/

import scala.util.Random
import viabilitree.viability._
//import java.io.File
import viabilitree.export._
import viabilitree.kdtree.Tree
import viabilitree.viability.kernel._

object PopulationViability extends App {
  val depth = 10
  val c = 0.5
  //  def stringToFile(s: String): better.files.File = File(s)
  /*
  val file: java.io.File = new java.io.File("testTest")
  Pop.run3(depth, file, umax)
  */
  val file: java.io.File = new java.io.File("experimentTime")
  Pop.runTest(depth, file, c)
}

object Pop {

  def run2(depth: Int) = {
    val population = Population()
    val rng = new Random(42)
    def a = 0.2
    def b = 3.0
    def c = 0.5
    def d = -2.0
    def e = 2.0

    val vk = KernelComputation(
      dynamic = population.dynamic,
      depth = depth,
      zone = Vector((a, b), (d, e)),
      controls = Vector(-0.5 to 0.5 by 0.02))

    val begin = System.currentTimeMillis()
    val (ak, steps) = approximate(vk, rng)
    // saveVTK2D(ak, s"/tmp/populationFINAL/population${steps}.vtk")
    val tps = (System.currentTimeMillis - begin)
    tps
  }

  def run1(depth: Int) = {
    val population = Population()
    val rng = new Random(42)
    def a = 0.2
    def b = 3.0
    def c = 0.5
    def d = -2.0
    def e = 2.0

    val vk = KernelComputation(
      dynamic = population.dynamic,
      depth = depth,
      zone = Vector((a, b), (d, e)),
      controls = Vector(-0.5 to 0.5 by 0.02))

    val begin = System.currentTimeMillis()
    val (ak, steps) = approximate(vk, rng)
    //    saveVTK2D(ak, s"/tmp/populationFINAL/population${steps}.vtk")
    val tps = (System.currentTimeMillis - begin)
    tps
  }

  def run(depth: Int, file: java.io.File, c: Double) = {
    val population = Population()
    val rng = new Random(42)
    def a = 0.2
    def b = 3.0
    def d = -2.0
    def e = 2.0

    val vk = KernelComputation(
      dynamic = population.dynamic,
      depth = depth,
      zone = Vector((a, b), (d, e)),
      //     controls = Vector(-0.5 to 0.5 by 0.02))
      controls = Vector(-c to c by 0.02))

    val begin = System.currentTimeMillis()
    val (ak, steps) = approximate(vk, rng)

    val f = file.toScala / s"${steps}depth${depth}.vtk"
    saveVTK2D(ak, f)
    println(volume(ak))
    val f2 = file.toScala / s"${steps}depth${depth}withControl${c}.txt"
    saveHyperRectangles(vk)(ak, f2)
    val f3 = file.toScala / s"${steps}depth${depth}withControl${c}.bin"
    save(ak, f3)
    val ak2 = load[Tree[KernelContent]](f3)
    println(volume(ak2))
    val tps = (System.currentTimeMillis - begin)
    tps
  }

  def runTest(depth: Int, file: java.io.File, c: Double) = {
    val population = Population()
    val rng = new Random(42)
    def a = 0.2
    def b = 3.0
    def d = -2.0
    def e = 2.0

    val vk = KernelComputation(
      dynamic = population.dynamic,
      depth = depth,
      zone = Vector((a, b), (d, e)),
      //     controls = Vector(-0.5 to 0.5 by 0.02))
      controls = Vector(-c to c by 0.02))

    val begin = System.currentTimeMillis()
    val (ak, steps) = approximate(vk, rng)

    println("nb of steps : ")
    println(steps)

    val tps = (System.currentTimeMillis - begin)
    tps

    println(tps)

    /*
    val f = file.toScala / s"${steps}depth${depth}.vtk"
    saveVTK2D(ak, f)
    println(volume(ak))
    val f2 = file.toScala / s"${steps}depth${depth}withControl${u_max}.txt"
    saveHyperRectangles(vk)(ak, f2)
    val f3 = file.toScala / s"${steps}depth${depth}withControl${u_max}.bin"
    save(ak,f3)
    val ak2 = load[Tree[KernelContent]](f3)
    println(volume(ak2))
    val tps = (System.currentTimeMillis - begin)
    tps
    */
  }

  /* def run(depth: Int, file: java.io.File, u_max: Double) = {
    import viabilitree.model.Dynamic

    val rng = new Random(42)
    def a = 0.2
    def b = 3.0
    def c = 0.5
    def d = -2.0
    def e = 2.0

    def populationDynamic(integrationStep: Double = 0.01, timeStep: Double = 0.1)(state: Vector[Double], control: Vector[Double]) = {
      def xDot(state: Vector[Double], t: Double) = state(1) * state(0)
      def yDot(state: Vector[Double], t: Double) = control(0)
      val dynamic = Dynamic(xDot, yDot)
      val res = dynamic.integrate(state.toArray, integrationStep, timeStep)
      res
    }

    val vk = KernelComputation(
      dynamic = populationDynamic,
      depth = depth,
      zone = Vector((a, b), (d, e)),
      controls = Vector(-u_max to u_max by 0.02))

    val (ak, steps) = approximate(vk, rng)

    println(volume(ak))

    // save viability kernel to files

    /*
    val f = file.toScala / s"${steps}depth${depth}.vtk"
    saveVTK2D(ak, f)
    val f2 = file.toScala / s"${steps}depth${depth}withControl${u_max}.txt"
    saveHyperRectangles(vk)(ak, f2)
    val f3 = file.toScala / s"${steps}depth${depth}withControl${u_max}.bin"
    save(ak,f3)
    // val ak2 = load[Tree[KernelContent]](f3)
    */

  }
*/
}

/*
object PopulationViability extends App
  with ViabilityKernel
  with ZoneInput
  with ZoneK
  with GridSampler {
  def a = 0.2
  def b = 3.0
  def c = 0.5
  def d = -2.0
  def e = 2.0

  def depth: Int = 18

  def zone = Seq((a, b), (d, e))
  //  def zone = Seq((a-0.01, b+0.01), (d-0.01, e+0.01))

  // def zone = Seq((0.2, 3.0), (-2.0, 2.0))
  //  def zone = Seq((0.0, 2.0), (0.0, 3.0))

  def dynamic(point: Point, control: Point) = Population(point, control)
  lazy val controls = (-0.5 to 0.5 by 0.02).map(Control(_))

  def dimension = 2

  def initialZone = zone

  implicit lazy val rng = new Random(42)

  //TODO replace when debug is over
  /*
  for {
    (b, s) <- apply.zipWithIndex
  } {
    println(s)
    b.saveVTK2D(Resource.fromFile(s"/tmp/population/populationGRID${depth}s$s.vtk"))
  }
*/

  val begin = System.currentTimeMillis()

  val listeResult = apply.zipWithIndex
  listeResult.foreach {
    case (b, s) =>
      //if (listeResult.hasNext && (s % 10 != 0)) println("on passe")
      //else {
      println(s"step $s")
//      if (s % 20 == 0 || !listeResult.hasNext) saveVTK2D(b, s"/tmp/population/population${s}.vtk")
      if (s % 20 == 0 || !listeResult.hasNext) saveVTK2D(b, s"/tmp/populationFINAL/population${s}.vtk")
    //}
  }

  println(System.currentTimeMillis - begin)

}
*/
