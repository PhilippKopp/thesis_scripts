for i in `seq 200 399`;
	do
		num=$(($i*5+0))
		~/scripts/analyse_AFLW2000_3DDFA_fittings.py $num &
		P1=$!

		num=$(($i*5+1))
		~/scripts/analyse_AFLW2000_3DDFA_fittings.py $num &
		P2=$!

		num=$(($i*5+2))
		~/scripts/analyse_AFLW2000_3DDFA_fittings.py $num &
		P3=$!

		num=$(($i*5+3))
		~/scripts/analyse_AFLW2000_3DDFA_fittings.py $num &
		P4=$!

		num=$(($i*5+4))
		~/scripts/analyse_AFLW2000_3DDFA_fittings.py $num &
		P5=$!

		wait $P1 $P2 $P3 $P4 $P5
        done    
