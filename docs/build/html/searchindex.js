Search.setIndex({docnames:["documentation/changelog","documentation/install","index","modules","motornet","motornet.nets","motornet.nets.callbacks","motornet.nets.layers","motornet.nets.losses","motornet.nets.models","motornet.plants","motornet.plants.muscles","motornet.plants.skeletons","motornet.task","motornet.utils","motornet.utils.plotor","tutorials/colab-tutorials","tutorials/testnb"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,nbsphinx:4,sphinx:56},filenames:["documentation\\changelog.md","documentation\\install.md","index.rst","modules.rst","motornet.rst","motornet.nets.rst","motornet.nets.callbacks.rst","motornet.nets.layers.rst","motornet.nets.losses.rst","motornet.nets.models.rst","motornet.plants.rst","motornet.plants.muscles.rst","motornet.plants.skeletons.rst","motornet.task.rst","motornet.utils.rst","motornet.utils.plotor.rst","tutorials\\colab-tutorials.md","tutorials\\testnb.ipynb"],objects:{"":[[4,0,0,"-","motornet"]],"motornet.nets":[[6,0,0,"-","callbacks"],[7,0,0,"-","layers"],[8,0,0,"-","losses"],[9,0,0,"-","models"]],"motornet.nets.callbacks":[[6,1,1,"","BatchLogger"],[6,1,1,"","BatchwiseLearningRateScheduler"],[6,1,1,"","TensorflowFix"],[6,1,1,"","TrainingPlotter"]],"motornet.nets.callbacks.BatchLogger":[[6,2,1,"","on_batch_end"],[6,2,1,"","on_train_begin"]],"motornet.nets.callbacks.BatchwiseLearningRateScheduler":[[6,2,1,"","on_batch_end"],[6,2,1,"","on_epoch_begin"]],"motornet.nets.callbacks.TensorflowFix":[[6,2,1,"","on_train_batch_end"],[6,2,1,"","on_train_begin"]],"motornet.nets.callbacks.TrainingPlotter":[[6,2,1,"","on_batch_end"],[6,2,1,"","on_train_begin"]],"motornet.nets.layers":[[7,1,1,"","GRUNetwork"],[7,1,1,"","Network"],[7,3,1,"","recttanh"]],"motornet.nets.layers.GRUNetwork":[[7,2,1,"","build"],[7,2,1,"","forward_pass"],[7,2,1,"","get_initial_state"],[7,2,1,"","get_save_config"]],"motornet.nets.layers.Network":[[7,2,1,"","call"],[7,2,1,"","forward_pass"],[7,2,1,"","from_config"],[7,2,1,"","get_base_config"],[7,2,1,"","get_base_initial_state"],[7,2,1,"","get_initial_state"],[7,2,1,"","get_save_config"]],"motornet.nets.losses":[[8,1,1,"","CompoundedLoss"],[8,1,1,"","L2ActivationLoss"],[8,1,1,"","L2ActivationMuscleVelLoss"],[8,1,1,"","L2Regularizer"],[8,1,1,"","L2xDxActivationLoss"],[8,1,1,"","L2xDxRegularizer"],[8,1,1,"","PositionLoss"],[8,1,1,"","RecurrentActivityRegularizer"]],"motornet.nets.models":[[9,1,1,"","DistalTeacher"],[9,1,1,"","MotorNetModel"]],"motornet.nets.models.DistalTeacher":[[9,2,1,"","from_config"],[9,2,1,"","get_config"],[9,2,1,"","save_model"],[9,2,1,"","train_step"]],"motornet.plants":[[11,0,0,"-","muscles"],[10,0,0,"-","plants"],[12,0,0,"-","skeletons"]],"motornet.plants.muscles":[[11,1,1,"","CompliantTendonHillMuscle"],[11,1,1,"","Muscle"],[11,1,1,"","ReluMuscle"],[11,1,1,"","RigidTendonHillMuscle"],[11,1,1,"","RigidTendonHillMuscleThelen"]],"motornet.plants.muscles.Muscle":[[11,2,1,"","activation_ode"],[11,2,1,"","build"],[11,2,1,"","get_initial_muscle_state"],[11,2,1,"","get_save_config"],[11,2,1,"","integrate"],[11,2,1,"","setattr"],[11,2,1,"","update_ode"]],"motornet.plants.muscles.RigidTendonHillMuscle":[[11,2,1,"","build"]],"motornet.plants.muscles.RigidTendonHillMuscleThelen":[[11,2,1,"","build"]],"motornet.plants.plants":[[10,1,1,"","CompliantTendonArm26"],[10,1,1,"","Plant"],[10,1,1,"","ReluPointMass24"],[10,1,1,"","RigidTendonArm26"]],"motornet.plants.plants.Plant":[[10,2,1,"","add_muscle"],[10,2,1,"","draw_fixed_states"],[10,2,1,"","draw_random_uniform_states"],[10,2,1,"","get_geometry"],[10,2,1,"","get_initial_state"],[10,2,1,"","get_muscle_cfg"],[10,2,1,"","get_save_config"],[10,2,1,"","integrate"],[10,2,1,"","integration_step"],[10,2,1,"","joint2cartesian"],[10,2,1,"","print_muscle_wrappings"],[10,2,1,"","setattr"],[10,2,1,"","state2target"],[10,2,1,"","update_ode"]],"motornet.plants.skeletons":[[12,1,1,"","PointMass"],[12,1,1,"","Skeleton"],[12,1,1,"","TwoDofArm"]],"motornet.plants.skeletons.PointMass":[[12,2,1,"","get_save_config"]],"motornet.plants.skeletons.Skeleton":[[12,2,1,"","build"],[12,2,1,"","clip_velocity"],[12,2,1,"","get_base_config"],[12,2,1,"","get_save_config"],[12,2,1,"","integrate"],[12,2,1,"","joint2cartesian"],[12,2,1,"","path2cartesian"],[12,2,1,"","setattr"],[12,2,1,"","update_ode"]],"motornet.plants.skeletons.TwoDofArm":[[12,2,1,"","get_save_config"]],"motornet.tasks":[[13,1,1,"","CentreOutReach"],[13,1,1,"","DelayedReach"],[13,1,1,"","RandomTargetReach"],[13,1,1,"","RandomTargetReachWithLoads"],[13,1,1,"","Task"]],"motornet.tasks.CentreOutReach":[[13,2,1,"","generate"]],"motornet.tasks.DelayedReach":[[13,2,1,"","generate"]],"motornet.tasks.RandomTargetReach":[[13,2,1,"","generate"]],"motornet.tasks.RandomTargetReachWithLoads":[[13,2,1,"","generate"]],"motornet.tasks.Task":[[13,2,1,"","add_loss"],[13,2,1,"","generate"],[13,2,1,"","get_attributes"],[13,2,1,"","get_initial_state"],[13,2,1,"","get_initial_state_layers"],[13,2,1,"","get_input_dict_layers"],[13,2,1,"","get_input_dim"],[13,2,1,"","get_losses"],[13,2,1,"","get_save_config"],[13,2,1,"","print_attributes"],[13,2,1,"","print_losses"],[13,2,1,"","set_training_params"]],"motornet.utils":[[14,0,0,"-","parallelizer"],[15,0,0,"-","plotor"]],"motornet.utils.plotor":[[15,3,1,"","animate_arm_trajectory"],[15,3,1,"","compute_limits"],[15,3,1,"","plot_2dof_arm_over_time"],[15,3,1,"","plot_pos_over_time"]],motornet:[[5,0,0,"-","nets"],[10,0,0,"-","plants"],[13,0,0,"-","tasks"],[14,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function"},terms:{"0":[6,7,9,10,11,12,13,15],"0002":10,"001":11,"00483":[10,11],"01":[10,11],"013193":12,"015":11,"020062":12,"05":[11,13],"051":10,"057":10,"1":[6,7,8,9,10,11,12,13,14,15,17],"10":[8,9,10,11,12,15],"100":6,"1000":12,"1025":8,"1033":8,"1038":8,"104":[10,11],"1115":[11,12],"1152":[10,11],"12":15,"1207":9,"125":11,"12661198":11,"135":10,"13633":12,"15":13,"1531112":11,"155":10,"16":9,"165":10,"18":8,"180496":12,"181479":12,"1992":9,"2":[6,7,8,9,10,12,14,15],"20":6,"2003":11,"2010":[10,11],"2013":12,"2015":8,"20884757":[10,11],"25":13,"25905111":12,"26":12,"2985":[10,11],"2d":[10,15],"3":[6,9,13,14],"307":9,"309":[10,12],"333":10,"354":9,"4":[10,12,14,15],"4042":8,"42872":6,"43":10,"45":13,"5":7,"50":13,"500":10,"534315":12,"6":[6,10,11,13],"640x480":15,"7":[9,11],"70":11,"8":[10,11,13],"82":10,"864572":12,"94":[10,11],"abstract":[7,13],"boolean":13,"case":[10,12],"catch":13,"class":[5,6,7,8,9,10,11,12,13],"default":[6,7,8,9,10,12,13],"do":[7,10,12,13],"float":[6,7,8,10,11,12,13,15],"function":[6,7,9,10,11,12,14,15],"import":13,"int":[7,10,11,12,15],"long":7,"new":[6,7,10,11,12],"null":10,"return":[6,7,9,10,11,12,13,15],"true":[6,13],"try":[12,13],"while":[7,10],A:[2,6,7,8,9,10,11,12,13,15],As:6,At:[7,13],By:[7,12],For:[6,7,8,9,10,11,12,13,15],If:[6,7,8,10,11,13,14],In:[5,6,9,11],Is:12,It:[7,9,11,13],Its:[10,15],Not:9,The:[6,7,8,9,10,11,12,13,14],Then:[10,12],There:[8,10,13],These:[7,10,11,13],To:[10,13],_:8,ab:8,abc:9,abid:7,about:[9,13],abov:[7,8,10],absolut:9,accord:[8,10,11],accuraci:9,achiev:6,across:12,act:7,activ:[6,7,8,11,13],activation_od:11,activity_weight:8,actual:13,actuat:11,ad:[10,13],adapt:8,add:[7,10,12,13,15],add_loss:13,add_muscl:[10,11],addit:[7,9,12,13],adjust:[6,11],adult:11,after:6,al:8,algorithm:9,alia:9,all:[7,10,11,12,13],allow:[6,9,10,11,12,14],alongsid:10,also:[10,12,13],altern:13,alwai:[9,10,11],amount:[10,11,14],an:[5,6,7,8,9,10,11,12,13,15],analys:14,angl:12,angular:13,angular_step:13,ani:[7,11,12,13],animate_arm_trajectori:15,anticip:13,anyth:7,api:9,api_doc:6,appli:[6,7,8,10,12,13],approxim:10,ar:[6,7,8,9,10,11,12,13],arg:[7,9],argument:[6,8,9,10,11,12,13],arm26:15,arm:[10,11,12,15],arm_anim:15,around:[10,11,13,15],arrai:[7,8,9,10,11,13,15],articl:11,ask:13,asm:12,assign:13,assigned_output:13,associ:10,assum:11,attach:[10,12],attr:13,attribut:[10,11,12,13],aug:12,auto:8,automat:[7,12],avail:13,ax:15,axi:8,backward:[6,7,9,10,12],base:[5,6,7,8,9,10,11,12,13],base_lay:7,batch:[6,7,10,11,13],batch_siz:[7,10,11,13],batchlogg:6,batchwiselearningrateschedul:6,becaus:[10,12],becom:15,been:6,befor:7,begin:6,behav:11,behaviour:6,being:[7,8,10,11,14,15],below:[7,12,13],between:[7,8,10,13],bia:9,bias:7,biolog:10,biomech:11,biomechan:[10,12,13],blob:[],bodi:[10,12],bone:[10,12],bool:[6,13],both:8,bound:[10,11,13],boundari:[10,12],bridg:5,brighter:15,build:[7,9,10,11,12,13,14,16],built:[10,11],calcul:[8,10,12],call:[6,7,9,10,11,12,13,14],callabl:13,callback:[3,4,9],callbacklist:9,callbal:13,can:[5,6,7,8,9,10,11,12,13,14],capabl:[7,9],cart_result:15,cartesian:[6,7,10,12],catch_trial_perc:13,cell:13,center:[7,10,12,13],centr:[10,13],central:[10,11],centreoutreach:13,chang:[10,11,12],changelog:2,churchland:8,classmethod:[7,9],clip:[11,12],clip_veloc:12,closest:10,cmap:15,code:13,cognit:9,colab:2,color:15,colormap:15,com:6,come:10,command:[7,10],commun:10,compar:[8,9],compat:[6,7,9,10,12,13],compil:9,complex:12,compliant:[10,11],complianttendonarm26:10,complianttendonhillmuscl:11,composit:8,compound:[8,13],compounded_loss:8,compoundedloss:8,comput:[7,10,11,12,13,15],compute_limit:15,conceptu:[5,9,12],concis:10,conf:12,config:[7,9],configur:[7,9,10,11,12,13,14,15],conjunct:10,connect:[7,9],consid:[9,12],consol:[9,14],constant:[8,11],constrain:10,contain:[5,6,7,9,10,11,12,13,14,15],content:[3,11,12],contract:11,contribut:[6,8,13],control:[6,7,8,9,10,12,16],convent:[7,8],convert:10,coordin:[10,12],core:14,corioli:12,corner:10,correspond:[6,8,10,11,13],cost:[10,11],counter:10,coupl:8,cpu:14,creat:[7,8,9,10,13],creation:7,cue:13,current:[6,10,12,13],custom:[6,8,9,10],custom_object:9,d:8,da:[8,10,11],darker:15,data:[6,9,15],data_util:13,dataset:15,dayth:0,de:[9,12],deactiv:11,dec:[10,11],declar:[8,9,11,13,14],deep:[2,7],defin:[7,8,9,10,11,13],deg:13,degre:[10,12,13,15],delai:[7,10,13],delay_rang:13,delayedreach:13,delp:12,demo:[],dens:7,depend:[10,13,14],deriv:[8,10,11,12,13],deriv_weight:[8,13],descend:11,desir:[9,10,13],detail:[6,7,8,9,10,11,12],detc2013:12,developp:11,deviat:[7,10],dg:11,dictionari:[6,7,9,10,11,12,13],dictionnari:12,didder:13,differenti:[10,11,12],dimens:[7,10,13,15],dimension:[10,11,12,13,15],direct:13,directli:11,directori:[9,14],discret:7,displai:6,distal:9,distalteach:[9,14],distanc:13,distribut:[7,10,13],document:[6,7,8,9,13],doe:[7,8,9,10,11,13,14],dof:12,doi:[8,9,10,11,12],done:[10,12],draw:10,draw_fixed_st:10,draw_random_uniform_st:10,drawn:[10,13],drive:[7,11],driven:11,dt:[8,10,11,12],dtype:7,dure:[6,8,9,10,13],dx:8,dynam:11,e:[7,10,12,13],each:[5,6,7,8,10,12,13],earli:15,earlier:15,effect:12,either:[6,10],elbow:[10,15],element:[10,11,12,13],empir:9,emploi:8,empti:6,enabl:9,end:6,endpoint:[10,12,13],endpoint_load:[10,12,13],energi:[10,11],eng:[11,12],engin:[7,9],ensur:[6,13],entri:[7,11,12],entrypoint:13,environ:10,epoch:6,epub:[10,11],equal:[10,11,12,13],equat:[10,11,12],equival:10,error:8,essenc:11,essenti:9,et:8,euler:[8,10,12],evalu:[8,10,11,12],even:6,everi:[6,7,10],evolv:12,exactli:12,exampl:[9,13],except:13,excit:[7,10,11],excitation_noise_sd:10,excitatori:7,execut:7,exist:7,expect:7,extens:[5,9],extra:11,extract:8,f:8,fals:13,featur:9,feb:11,fed:8,feedback:[7,10],feel:6,fetch:8,figur:15,file:[9,14],find:8,first:[7,10,12,13,14],fit:[10,13],fix:6,fixat:[10,12],float32:7,fo:10,follow:[7,10,11,12],forc:[8,10,11,12],format:[10,11,13],formul:11,forward:[7,9,10],forward_pass:7,four:[5,14],frame:10,free:6,freedom:[10,12,13,15],frequent:6,from:[6,7,8,9,10,11,12,13,14,15],from_config:[7,9],full:[7,11,12],further:12,g:12,gaussian:[7,10],gener:[10,12,13,14],geometri:[7,10,11],geometry_st:[10,11],get:[7,9,10,11,12,13],get_attribut:13,get_base_config:[7,12],get_base_initial_st:7,get_config:[7,9],get_geometri:10,get_initial_muscle_st:11,get_initial_st:[7,10,13],get_initial_state_lay:13,get_input_dict_lay:13,get_input_dim:13,get_loss:13,get_muscle_cfg:10,get_save_config:[7,10,11,12,13],github:6,githubtocolab:[],give:[8,10,13,15],given:[6,10,11,12,13,15],global:7,go:13,go_cue_rang:13,graviti:12,gribbl:[10,11],group:9,gru:[7,8],gru_regular:8,grunetwork:[7,13],guid:9,h:8,ha:[6,13,15],half:12,handl:[7,9,10,13,15],have:[10,11,12,15],held:[10,11,12,13],hello:17,help:6,henc:8,here:[6,9,12],hidden:[7,8],hidden_noise_sd:7,hill:11,histori:6,how:[2,6,9,13,14,16],howev:12,http:[6,8,9],hyperbol:7,i1:[10,12],i2:[10,12],i:[7,10,13],ident:12,ignor:[6,13],implement:[5,6,7,8,9,10,11,12],improv:14,includ:[6,8,9,13],index:[2,6,8,10],indic:[6,8,10,12,13],inertia:12,infer:[7,9,10,11,13],inform:[6,7,9,10,13,14],inher:7,inherit:11,initi:[0,6,7,9,10,11,12,13],initial_joint_st:13,initial_st:13,input:[6,7,8,9,10,11,12,13,14],input_dim:[11,12],input_shap:7,insensit:[10,12],instal:2,instanc:[7,8,9,10,11,12,13,15],instanti:[7,9],instead:[6,9,10,12,13],integ:[6,7,10,11,12,13,14,15],integr:[10,11,12],integration_method:[10,12],integration_step:[10,12],interrupt:6,introduct:9,invok:7,ipsum:1,ipynb:[],isometr:[8,11],issu:6,item:13,iter:14,its:[7,8,9,10,11],itself:[5,8,9,10],j:[10,11],jd:[10,11],jn:[10,11],joint2cartesian:[10,12],joint:[7,10,12,13],joint_load:10,joint_posit:15,joint_stat:[10,12,13,15],jordan:9,json:[9,14],jul:9,k:[7,13],kaufman:8,keep:6,kei:[7,10,13],kept:6,kera:[6,7,8,9,10,13],kernel:7,kernel_regular:7,keyword:11,kg:[10,12],kistemak:[10,11],kutta4:[10,12],kutta:[10,12],kwarg:[7,9,10,11,12,13,15],l1:[8,10,12],l1g:[10,12],l2:[8,10,12,13],l2_activ:8,l2_activation_muscle_vel:8,l2_regular:8,l2_xdx_activ:8,l2activationloss:8,l2activationmusclevelloss:8,l2g:[10,12],l2regular:8,l2xdxactivationloss:8,l2xdxregular:8,label:[8,13],larg:14,last:[6,7],later:15,layer:[3,4,8,9,13],learn:[6,9,13],learningrateschedul:6,least:[7,13],length:[10,11,12],level:[10,12],lighter:15,like:[9,12,14],limit:[10,13,15],line:15,linear:[7,11],linewidth:15,list:[7,8,9,10,11,12,13,15],load:[10,12,13],log:6,logic:[7,9,13],loop:7,lorem:1,loss:[3,4,6,9,13],loss_weight:[8,13],losses_util:8,lossfunctionwrapp:8,lower:[10,11,12,13,15],lump:10,m1:[10,12],m2:[10,12],m:[8,10,11,12,13],ma:12,made:[6,10],mai:13,main:10,mani:[6,10,11,13,14],map:[7,10,13],margin:15,mass:[10,12],master:[],match:[10,11,12,13],matplotlib:15,max_iso_forc:8,max_isometic_forc:10,max_isometric_forc:[10,11],maximum:[8,10,11],mean:[9,11,15],mechan:11,messag:6,method:[6,7,8,9,10,11,12,13],metric:[6,9],mi:9,min_activ:11,minim:[10,11],minimum:11,ministep:7,miss:10,mn:13,mode:[6,14],model:[3,4,6,7,8,10,11,12,13,14],modul:[2,3,6,7,8,15],moment:[10,12],monitor:6,month:0,more:[6,8,9,10,12,13,14],motor:[7,8,10],motornet:3,motornetmodel:9,move:12,movement:[2,7,9,10,11,13,15],mp4:15,ms:7,multipl:10,muscl:[3,4,6,7,8,12,13,16],muscle_input:10,muscle_m:10,muscle_st:[10,11],muscle_typ:[7,10],musculotendon:12,must:8,my_model_config:9,n:[10,11,12,13],n_batch:[11,12,15],n_dim:15,n_dof:12,n_fixation_point:12,n_hidden_lay:7,n_ministep:7,n_muscl:[7,11],n_state:11,n_timestep:[10,11,13,15],n_unit:[7,13],name:[7,8,9,10,11,12,13],nat:8,naturalist:8,ndarrai:[13,15],necessari:7,need:[7,10,11],neither:13,nervou:[10,11],nest:[9,10],net:[3,4,13,14],network:[2,6,7,8,9,10,13,16],neural:[2,7,8],neurophysiol:[10,11],neurosci:8,next:10,nightli:6,nn:8,nois:[7,10],non:[7,13],none:[6,7,9,10,13],nor:[7,9],normal:[7,8],normalized_slack_muscle_length:11,note:[10,11,12],notebook:2,notimplementederror:7,now:6,np:8,number:[6,7,10,11,12,13,14],numer:[10,11,12],numpi:[13,15],obei:[7,13],object:[5,6,7,8,9,10,11,12,13,14,15,16],obtain:[7,10,11],occasion:12,occur:[6,13],od:12,off:5,older:11,oliviercodol:[],on_batch_end:6,on_epoch_begin:6,on_train_batch_end:[6,9],on_train_begin:6,on_training_end:6,one:[7,8,9,10,11,12,13],ongo:6,onli:[6,7,9,11],onlin:9,onto:13,oper:7,oppos:13,optim:[11,13],optimal_muscle_length:11,option:[6,7,10,11,12,13],order:[7,8,10,15],ordinari:[10,11,12],org:[6,8,9],origin:[9,10,11],orthogon:10,other:[5,7,10,13],out:13,output:[6,7,8,9,10,11,12,13],output_bias_initi:7,output_dim:[11,12],output_kernel_initi:7,output_nam:13,outsid:10,over:[7,10,14,15],overrid:[7,9],overridden:13,overwrit:6,overwritten:[7,12],own:[6,10],packag:6,page:2,parallel:[3,4],param:14,paramet:[6,7,8,9,10,11,12,13,14,15],parent:[6,7,10,11,12,13],part:[12,13,15],particular:11,particularli:14,pass:[6,7,8,9,10,11,12,13,14],passiv:11,past:[11,12],path2cartesian:12,path:[9,10,12],path_coordin:[10,12],path_fixation_bodi:[10,12],path_nam:15,penal:8,penalti:8,per:[7,13],percentag:13,perfect:9,perform:[2,6,7,9,10,11,12,13],pertain:10,pl:[10,11],placehold:9,planar:[12,15],plant:[3,4,5,7,8,9,13,15,16],plot:[6,8,13,15],plot_2dof_arm_over_tim:15,plot_freq:6,plot_loss:6,plot_n_t:6,plot_pos_over_tim:15,plot_trial:6,plotor:[3,4],pmc4404026:12,pmcid:12,pmid:[10,11,12],po:12,point:[10,12],point_mass:12,pointmass:12,polynomi:10,pos_lower_bound:[10,12],pos_upper_bound:[10,12],posit:[6,8,10,11,12,13,15],positionloss:8,possibl:10,potenti:[6,7],power:2,pre:[10,11,13],present:13,previou:6,print:[8,10,13,17],print_attribut:13,print_loss:13,print_muscle_wrap:10,proc:12,proce:10,procedur:7,process:7,produc:[9,10,11,13,14],product:8,propag:9,properti:[10,12],proport:15,propos:8,propriocept:[7,10],proprioceptive_delai:10,proprioceptive_noise_sd:7,provid:[6,7,8,10,11,13],pull:11,purpos:[5,9,13],push:11,python:[2,6,7,8,9,13,14],qualifi:10,quiet:6,rais:[7,10,13,14],random:[7,10,13],randomli:[10,13],randomtargetreach:13,randomtargetreachwithload:13,rang:[13,15],rate:6,reach:[2,13,15],reaching_dist:13,readabl:[10,13],realiti:10,receiv:10,recommend:10,recomput:9,rectifi:[7,11],recttanh:7,recurr:[7,8],recurrent_act:8,recurrent_regular:7,recurrent_weight:8,recurrentactivityregular:8,reduce_mean:8,reduct:8,reductionv2:8,redund:7,refer:[6,8,9,10,11,12],regardless:11,regular:[7,8],relat:13,releas:0,reli:[10,11],relumuscl:[10,11],relupointmass24:[10,13],remov:8,repres:[7,10,15],represent:9,reproduc:9,requir:[10,13],respect:[10,12,15],result:12,retriv:13,return_sequ:13,revers:[7,9],rigid:11,rigidtendonarm26:10,rigidtendonhillmuscl:11,rigidtendonhillmusclethelen:11,rk4:[10,12],rnn:13,round:10,routin:[6,10],rumelhart:9,run:[14,16],rung:[10,12],rungekutta4:[10,12],s15516709cog1603_1:9,s:[6,7,8,9,10,11,12,13,14,15],same:[7,9,10,11,12,13],sampl:9,save:[6,7,8,9,12,13,14],save_model:[9,14],scalar:8,scale:[8,11],schedul:6,scienc:9,script:14,scructur:11,search:2,sec:[10,12,13],second:[11,12,14],section:9,see:[6,7,8,9,10,12,13,14],segment:15,sep:[10,11],sequenc:13,seri:[5,10],serv:[9,12],session:[6,15],set:[6,7,10,11,12,13,15],set_training_param:13,set_weight:[7,9],setattr:[10,11,12],seth:12,sever:[7,8,11],shape:13,sherman:12,should:[6,7,9,10,11,12,13,14,15],shoulder:[10,15],signal:[7,10,11],significantli:14,simpl:[11,12],simul:[6,7,11,15],sinc:[9,11,15],singl:[7,8,10,11,12,13],size:[7,8,10,11,12,13,15],skeleton:[3,4,11,13,15],sl:12,so:[7,9,10,11,12],solut:8,some:[6,9,10,12],space:[10,12,13,15],space_dim:[12,13],specif:[7,10,12,13],specifi:[10,12,13],speed:14,split:8,stand:10,standard:[7,10],start:[6,11,13],start_posit:13,state2target:10,state:[7,8,10,11,12,13],state_deriv:[10,11,12],state_dim:12,state_f:13,state_i:13,step:[7,9,10,11,12],stochast:10,str:[8,10,11,12,13,15],strictli:10,string:[7,8,9,10,11,12,13,14,15],structur:[7,9],subclass:[5,6,7,8,9,10,11,12,13],subloss:8,submodul:5,subpackag:3,subplot:6,subsequ:6,sum:8,supervis:9,support:15,sussillo:8,synthet:9,system:[10,11],take:[6,7,10,11,12,14],taken:12,tangent:7,target:[9,10,13],task:[3,6,9,14],tau_activ:11,tau_deactiv:11,teacher:9,tech:12,technic:10,tendon:[10,11],tendon_length:11,tensoflow:8,tensor:[7,9,10,11,12,13],tensorflow:[2,5,6,7,8,9,10,13],tensorflowfix:6,tensorshap:7,term:10,termin:14,test:[2,6,13,14],tf:[6,7,8,9,13],th:13,than:11,thei:[6,7,8,10],thelen:11,them:10,therefor:[7,11,13],thi:[5,6,7,8,9,10,11,12,13,14,15],third:15,those:[11,13],through:[7,9,10],tile:10,time:[6,7,10,11,12,13,15],timestep:[6,7,8,10,11,12,13],toggl:6,toolbox:2,top:12,torqu:12,total:[6,13,14],track:6,train:[2,6,7,8,9,10,13,14,15],train_step:9,trainingplott:6,trajectori:15,transform:12,trial:[6,13],tupl:[8,10,12,13],tutori:9,two:[10,12,13,15],two_dof_arm:12,twodofarm:[10,12,15],type:[10,11,13,14],typeerror:10,typic:[5,7,9,10,12],uniform:[10,13],union:[8,10,12,13],unit:[6,7,8],unlik:11,up:[10,14],updat:[6,7,10,11],update_od:[10,11,12],upper:[10,11,12,13,15],ur:12,us:[6,7,8,9,10,11,12,13,14,15],user:[6,7,8,12,14],usual:[9,10,11,12,13],util:[3,4,8,13],valid:13,valu:[6,7,8,9,10,11,12,13,15],valueerror:[10,13,14],vari:10,variabl:[7,8],variou:[14,15],vector:[10,11,12],vel:12,vel_lower_bound:[10,12],vel_upper_bound:[10,12],veloc:[6,8,10,12,15],verbos:6,veri:[7,13],version:[10,11],via:[10,11],viridi:15,virtual:7,viscos:12,visit:6,visual:[7,10],visual_delai:10,visual_noise_sd:7,wa:[6,8,10,13],wait:13,want:[7,9,12,13],we:[8,9,10,12,15],weight:[6,7,8,9,13],well:[6,9,12],what:[6,9,10,12,13],when:[9,13],where:13,whether:[6,13,14],which:[6,7,9,10,11,12,13],whose:[7,9],width:15,window:13,within:10,without:[9,13],wong:[10,11],word:5,work:16,world:17,worldspac:[10,12,13],would:[8,9,10,12,15],wrap:[8,10,11],wrapper:[11,12],www:[6,9],x:[7,8,10,15],xdx:8,xp:8,xy:12,y:[8,15],y_true:13,year:0,yield:11,yp:8,zero:10},titles:["Changelog","How to install","MotorNet","&lt;no title&gt;","MotorNet","motornet.nets","motornet.nets.callbacks","motornet.nets.layers","motornet.nets.losses","motornet.nets.models","motornet.plants","motornet.plants.muscles","motornet.plants.skeletons","motornet.task","motornet.utils","motornet.utils.plotor","COLAB tutorials","Test notebook"],titleterms:{"0":0,"1":0,"4":0,"do":1,Then:1,bar:1,callback:[5,6],changelog:0,colab:16,content:[4,5,10,14],document:2,font:0,foo:1,how:1,indic:2,instal:1,layer:[5,7],loss:[5,8],manual:2,model:[5,9],modul:[4,5,10,14],motornet:[2,4,5,6,7,8,9,10,11,12,13,14,15],muscl:[10,11],net:[5,6,7,8,9],notebook:17,parallel:14,plant:[10,11,12],plotor:[14,15],refer:2,size:0,skeleton:[10,12],subpackag:4,tabl:2,task:[4,13],test:17,tutori:[2,16],util:[14,15],version:0}})