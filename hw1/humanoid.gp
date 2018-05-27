set terminal png
set output "humanoid_bc.png"

set datafile separator '|'
set logscale x
set xlabel "training data rollouts"
set ylabel "mean reward"

plot "<grep '|' bc_rollouts.log" using 7:3:4 with yerrorbars lw 3 title 'Humanoid', '' using 7:3 with lines lw 2 title ''
