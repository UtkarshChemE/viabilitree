package viabilitree.approximation.example.raz13

/*
 * Copyright (C) 10/10/13 Isabelle Alvarez
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

import viabilitree.model._
import math._

case class RAZ13(
  integrationStep: Double = 0.01,
  timeStep: Double = 0.1,
  Tm: Double = 2.0,
  A2: Double = 0.1,
  b: Double = 1.0,
  C: Double = 2.0,
  A3: Double = 1.0,
  M: Double = 5.0,
  a3: Double = 2.0,
  a2: Double = 0.0,
  a1: Double = 0.0,
  a0: Double = 1.0,
  v_m: Double = 0.8) {

  /* PARAMETRES
  M flood size for which impact is half the max (1/2)
  A3 must be <= 1
  Damage parameters a3, a2, a1, a0
   */
  /* ON NE PEUT PAS FAIRE COMME CA
  def dynamic2(state: Vector[Double], control: Vector[Double]) = {
    val A2bis = min((1-A2)/2,0.2)
    // A2bis dans ]0,1-A2[
    def alphaDot(state: Vector[Double], t: Double) =
      - A1 * state(0) + (A2 + A2bis * state(0)) * state(0) * (1-state(0))
    def wDot(state: Vector[Double], t: Double) = b - C * control(0)

    val dynamic = Dynamic(alphaDot, wDot)
    dynamic.integrate(state.toArray, integrationStep, timeStep)
  }

def dynamic(state: Vector[Double], control: Vector[Double]) = {
  dynamic1(state,control)
}
*/
  def vMax = 5.0
  def A1 = log(2) / Tm
  // A1 peut valoir en fait ln(2)/TM, ie en TM alpha aura perdu la moitié de sa valeur initiale
  def dynamic(state: Vector[Double], control: Vector[Double]) = {
    def alphaDot(state: Vector[Double], t: Double) =
      -A1 * state(0) + A2 * state(0) * (1 - state(0)) * control(0)
    def wDot(state: Vector[Double], t: Double) = b - C * control(0)

    val dynamic = Dynamic(alphaDot, wDot)
    dynamic.integrate(state.toArray, integrationStep, timeStep)
  }

  def d_1(alpha: Double, s: Double): Double = {
    a3 * s * s * s + a2 * s * s + a1 * s
  }

  def d_2(alpha: Double, s: Double): Double = {
    s >= v_m match {
      case false => 0.0
      case true => a0 * (s - v_m) * (s - v_m) * (s - v_m)
    }
  }
  def damage(alpha: Double, s: Double): Double = {
    (1 - alpha) * d_1(alpha, s) + alpha * d_2(alpha, s)
  }
  // a0*(1-alpha)*s

  def perturbation(state: Vector[Double], s: Double) = {
    def alphaDelta(state: Vector[Double], s: Double) = A3 * (1 - state(0)) * (s / (M + s))
    def wDelta(state: Vector[Double], s: Double) = -damage(state(0), s)
    (alphaDelta(state, s), wDelta(state, s))
  }

  def jump(state: Vector[Double], s: Double) = {
    val (alphaDelta, wDelta) = perturbation(state, s)
    Vector(state(0) + alphaDelta, state(1) + wDelta)
  }
  // pour avoir un vecteur en sortie
  /*
  def jump(state:Vector[Double], s: Double) = {
    val (alphaDelta,wDelta) = perturbation(state,s)
    (state(0) + alphaDelta, state(1) + wDelta )
  }
*/

  // On a besoin d'une soft_appartenance à un noyau qui tienne compte de la manière dont on sort de l'ensemble
  // TODO fix pb when some states are outside wLim (ex. v=3) outOfMemoryError

  // true if the present point is in the kernel and if its image by the perturbation is also in the kernel OR if it is out the upper limit
  def softJump(state: Vector[Double], jumpV: Vector[Double] => Vector[Double],
    viableSet: viabilitree.kdtree.Tree[viabilitree.viability.kernel.KernelContent],
    viabProblem: viabilitree.viability.kernel.KernelComputation): Boolean = {
    val jumpState = jumpV(state)
    val zoneLim = viabProblem.zone
    val wLim = zoneLim.region(1).max
    (viableSet.contains(viabilitree.viability.kernel.KernelContent.label.get, state) &&
      (viableSet.contains(viabilitree.viability.kernel.KernelContent.label.get, jumpState)) ||
      jumpState(1) >= wLim)
  }

  // ici c'est différent il faut être pragmatique
  // inverse i.e. valeurs desquelles on est parti avant la perturbation s
  def inverseJumpDirect(state: Vector[Double], s: Double) = {
    val (alphaDirect, wDirect) = inversePerturbation(state, s)
    Vector(alphaDirect, wDirect)
  }

  def inversePerturbation(state: Vector[Double], s: Double) = {
    val Aa = s*A3/(M+s)
    def alphaDirect(state: Vector[Double], s: Double) = (state(0) - Aa)/(1-Aa)
    def wDirect(state: Vector[Double], s: Double) = state(1) + damage(alphaDirect(state, s),s)
    (alphaDirect(state, s), wDirect(state, s))
  }

  // true if the present point is the image of a kernel point by a perturbation (jumpV)
  def softInverseJump(state: Vector[Double], jumpV: Vector[Double] => Vector[Double],
               viableSet: viabilitree.kdtree.Tree[viabilitree.viability.kernel.KernelContent],
               viabProblem: viabilitree.viability.kernel.KernelComputation): Boolean = {
    val jumpState = jumpV(state)
    (viableSet.contains(viabilitree.viability.kernel.KernelContent.label.get, jumpState))
  }


}

