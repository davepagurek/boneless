# Locomotion of Boneless Creatures with Distributed Control

We are interested in exploring how plausible it is to control creatures whose bodies contain muscle and no bone.

## Optimization methods

We optimized policies using a genetic algorithm (GA) and using Proximal Policy Optimization (PPO).

### Genetic optimization
<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/davepagurek/boneless/master/gifs/ScreenCapture_16-04-2020%202.18.31%20PM.gif" /></td>
    <td><img src="https://raw.githubusercontent.com/davepagurek/boneless/master/gifs/ScreenCapture_15-04-2020%204.56.06%20PM.gif" /></td>
  </tr>
  <tr>
    <td><em>Tetrapus with high springiness</em></td>
    <td><em>Tetrapus with low springiness</em></td>
  </tr>
</table>

### Proximal Policy Optimization
<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/davepagurek/boneless/master/gifs/ppo_Trim.gif" /></td>
  </tr>
  <tr>
    <td><em>PPO creates stochastic policies, which are<br />not as effective as those found using GA.</em></td>
  </tr>
</table>

## Control Schemes

We wanted to compare how policy effectiveness changes when you go from a single controller with full state knowledge to a set of distributed controllers, with only local state knowledge and the ability to pass information to neighbouring controllers.

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/davepagurek/boneless/master/gifs/skinnyworm%20global%20gen%2050.gif" /></td>
    <td><img src="https://raw.githubusercontent.com/davepagurek/boneless/master/gifs/skinnyworm%20with%20comm%20gen%2050.gif" /></td>
    <td><img src="https://raw.githubusercontent.com/davepagurek/boneless/master/gifs/skinnyworm%20without%20comm%20gen%2050.gif" /></td>
  </tr>
  <tr>
    <td><em>Worm with global controller</em></td>
    <td><em>Worm with local, communicating controllers</em></td>
    <td><em>Worm with local controllers but no communication</em></td>
  </tr>
</table>

## Extras

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/davepagurek/boneless/master/gifs/horse%2005%20100th%20gen.gif" /></td>
  </tr>
  <tr>
    <td><em>GA can optimize quadrupeds too!</em></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/davepagurek/boneless/master/gifs/textured.gif" /></td>
  </tr>
  <tr>
    <td><em>When they play your song at the club</em></td>
  </tr>
</table>
