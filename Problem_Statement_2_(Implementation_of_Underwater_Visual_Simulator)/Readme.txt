1) Install ubuntu 18.04 as it is compatible only with that. I have tried in Ubuntu 22 and 20 but it failed.

2) Follow the documentation more than other Chatgpts, Grok etc. I have mentioned the documentation links in my report.

3) Type the below following commands in Ubuntu terminal.

4) sudo apt-get update
   sudo apt-get install ros-melodic-uwsim

5) In one terminal, run 
   roscore
   In another terminal, run
   rosrun uwsim uwsim

6) In order to make vehicle move etc,
   rosrun uwsim setVehicleVelocity /dataNavigator 0.2 0 0 0 0 0

7) to twist and to do more functions,
   rosrun uwsim setVehicleTwist /g500/twist 0.2 0 0 0 0 0
   rosrun uwsim setVehiclePose /g500/pose 1.0 0 0 0 0 0

More commands in Documentation !

