# $1, cuda device
# $2, Carla port
# $3, exp_name
# $4, repeat
# $5, agent config
# $6, part, 1s-5s

collab_method_list=("single" "late" "early" "codriving" "fcooper" "v2xvit")
collab_flag=true # false
agent_version=''

# for item in "${collab_method_list[@]}"; do
#     if [ "$item" == "$5" ]; then
#         collab_flag=true
#         break
#     fi
# done

export PART=${6:-1}

if [ $PART == "1s" ];  
then {
    route_list=(2	3	5	8	10	13	18	31	135	146	147	161	300	310	326	327)
    scen_config='_1'
    if $collab_flag; then
        agent_version='_8_10'
    fi
}
elif [ $PART == "2s" ];
then {
    route_list=(2	3	7	12	19	22	24	25	27	136	138	161	300	320	321	322	328	329)
    scen_config='_2'
    if $collab_flag; then
        agent_version='_5_10'
    fi
}
elif [ $PART == "3s" ];
then {
    route_list=(7	8	16	31	300	301	305	310	311	313	314	316	320	324	329	330	331)
    scen_config='_3'
    if $collab_flag; then
        agent_version='_5_10'
    fi
}
elif [ $PART == "4s" ];
then {
    route_list=(1	8	9	12	14	15	16	28	140	145	160	311	312	317	321	323	328	331)
    scen_config='_4'
    if $collab_flag; then
        agent_version='_8_10'
    fi
}
elif [ $PART == "5s" ];
then {
    route_list=(1	6	7	8	18	20	28	31	142	145	146	157	306	310	315	318)
    scen_config='_5'
    if $collab_flag; then
        agent_version='_5_50'
    fi
}
else {
    echo "Undefined PART yet!"
}
fi

for route_idx in ${route_list[*]}
do 
    echo route_idx $route_idx
    echo 'agent' "$5$agent_version" 'scenario' $scen_config
    CUDA_VISIBLE_DEVICES=$1 ./scripts/eval_driving_e2e.sh $route_idx $2 $3 $4$scen_config$agent_version "$5$agent_version" $scen_config
done