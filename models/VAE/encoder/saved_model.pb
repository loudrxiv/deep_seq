??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.22v2.6.1-9-gc2363d6d0258??
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ * 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:@ *
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
: *
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:?*
dtype0
|
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv1d/kernel
u
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*$
_output_shapes
:??*
dtype0
o
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv1d/bias
h
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:?*
dtype0

conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@* 
shared_nameconv1d_1/kernel
x
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*#
_output_shapes
:?@*
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:@*
dtype0
?
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
?
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
?
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
u

el1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
el1/kernel
n
el1/kernel/Read/ReadVariableOpReadVariableOp
el1/kernel*#
_output_shapes
:?*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:?*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:?*
dtype0
?
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
?
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
?
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0

NoOpNoOp
?7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?6
value?6B?6 B?6
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
 
y

layer_with_weights-0

layer-0
trainable_variables
regularization_losses
	variables
	keras_api
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
V
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
10
11
 
?
+0
!1
"2
,3
-4
#5
$6
%7
&8
.9
/10
'11
(12
)13
*14
015
116
17
18
?
2layer_regularization_losses
trainable_variables
3non_trainable_variables
regularization_losses

4layers
	variables
5layer_metrics
6metrics
 
^

+kernel
7trainable_variables
8regularization_losses
9	variables
:	keras_api
 
 

+0
?
;layer_regularization_losses
trainable_variables
<non_trainable_variables
regularization_losses

=layers
	variables
>layer_metrics
?metrics
R
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
?
Daxis
	!gamma
"beta
,moving_mean
-moving_variance
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
h

#kernel
$bias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
R
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
?
Qaxis
	%gamma
&beta
.moving_mean
/moving_variance
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
h

'kernel
(bias
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
R
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
?
^axis
	)gamma
*beta
0moving_mean
1moving_variance
_trainable_variables
`regularization_losses
a	variables
b	keras_api
F
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
 
v
!0
"1
,2
-3
#4
$5
%6
&7
.8
/9
'10
(11
)12
*13
014
115
?
clayer_regularization_losses
trainable_variables
dnon_trainable_variables
regularization_losses

elayers
	variables
flayer_metrics
gmetrics
[Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
hlayer_regularization_losses
trainable_variables
inon_trainable_variables
regularization_losses

jlayers
	variables
klayer_metrics
lmetrics
_]
VARIABLE_VALUEbatch_normalization/gamma0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEbatch_normalization/beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv1d/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv1d/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_1/gamma0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_1/beta0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1d_1/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv1d_1/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization_2/gamma0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization_2/beta0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
el1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/3/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!batch_normalization_1/moving_mean&variables/9/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/10/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_2/moving_mean'variables/15/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_2/moving_variance'variables/16/.ATTRIBUTES/VARIABLE_VALUE
 
1
+0
,1
-2
.3
/4
05
16

0
1
2
3
 
 
 
 

+0
?
mlayer_regularization_losses
7trainable_variables
nnon_trainable_variables
8regularization_losses

olayers
9	variables
player_metrics
qmetrics
 

+0


0
 
 
 
 
 
?
rlayer_regularization_losses
@trainable_variables
snon_trainable_variables
Aregularization_losses

tlayers
B	variables
ulayer_metrics
vmetrics
 

!0
"1
 

!0
"1
,2
-3
?
wlayer_regularization_losses
Etrainable_variables
xnon_trainable_variables
Fregularization_losses

ylayers
G	variables
zlayer_metrics
{metrics

#0
$1
 

#0
$1
?
|layer_regularization_losses
Itrainable_variables
}non_trainable_variables
Jregularization_losses

~layers
K	variables
layer_metrics
?metrics
 
 
 
?
 ?layer_regularization_losses
Mtrainable_variables
?non_trainable_variables
Nregularization_losses
?layers
O	variables
?layer_metrics
?metrics
 

%0
&1
 

%0
&1
.2
/3
?
 ?layer_regularization_losses
Rtrainable_variables
?non_trainable_variables
Sregularization_losses
?layers
T	variables
?layer_metrics
?metrics

'0
(1
 

'0
(1
?
 ?layer_regularization_losses
Vtrainable_variables
?non_trainable_variables
Wregularization_losses
?layers
X	variables
?layer_metrics
?metrics
 
 
 
?
 ?layer_regularization_losses
Ztrainable_variables
?non_trainable_variables
[regularization_losses
?layers
\	variables
?layer_metrics
?metrics
 

)0
*1
 

)0
*1
02
13
?
 ?layer_regularization_losses
_trainable_variables
?non_trainable_variables
`regularization_losses
?layers
a	variables
?layer_metrics
?metrics
 
*
,0
-1
.2
/3
04
15
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 

+0
 
 
 
 
 
 
 
 
 

,0
-1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

.0
/1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

00
11
 
 
 
?
serving_default_iptPlaceholder*4
_output_shapes"
 :??????????????????*
dtype0*)
shape :??????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_ipt
el1/kernel#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betaconv1d/kernelconv1d/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betaconv1d_1/kernelconv1d_1/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/betaconv1d_2/kernelconv1d_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_90484
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOpel1/kernel/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOpConst* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_91642
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d_2/kernelconv1d_2/biasbatch_normalization/gammabatch_normalization/betaconv1d/kernelconv1d/biasbatch_normalization_1/gammabatch_normalization_1/betaconv1d_1/kernelconv1d_1/biasbatch_normalization_2/gammabatch_normalization_2/beta
el1/kernelbatch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_91709??
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_89721

inputsB
+conv1d_expanddims_1_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2	
BiasAddx
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_91344

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?+
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_89373

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_90117

inputsA
+conv1d_expanddims_1_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? *
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :?????????????????? *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2	
BiasAddx
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
?
+__inference_embed_seq_2_layer_call_fn_89971
max_pooling1d_input
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	? 
	unknown_9:?@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmax_pooling1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_898992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:???????????????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
5
_output_shapes#
!:???????????????????
-
_user_specified_namemax_pooling1d_input
?
?
+__inference_embed_seq_2_layer_call_fn_90936

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	? 
	unknown_9:?@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_898992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:???????????????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
)__inference_embed_seq_layer_call_fn_90570

inputs
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?!
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?!

unknown_10:?@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16:@ 

unknown_17: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? */
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_embed_seq_layer_call_and_return_conditional_losses_902652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_2_layer_call_fn_91495

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_895032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
??
?
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_91152

inputsJ
;batch_normalization_assignmovingavg_readvariableop_resource:	?L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	?H
9batch_normalization_batchnorm_mul_readvariableop_resource:	?D
5batch_normalization_batchnorm_readvariableop_resource:	?J
2conv1d_conv1d_expanddims_1_readvariableop_resource:??5
&conv1d_biasadd_readvariableop_resource:	?L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	?N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	?F
7batch_normalization_1_batchnorm_readvariableop_resource:	?K
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:?@6
(conv1d_1_biasadd_readvariableop_resource:@K
=batch_normalization_2_assignmovingavg_readvariableop_resource:@M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@
identity??#batch_normalization/AssignMovingAvg?2batch_normalization/AssignMovingAvg/ReadVariableOp?%batch_normalization/AssignMovingAvg_1?4batch_normalization/AssignMovingAvg_1/ReadVariableOp?,batch_normalization/batchnorm/ReadVariableOp?0batch_normalization/batchnorm/mul/ReadVariableOp?%batch_normalization_1/AssignMovingAvg?4batch_normalization_1/AssignMovingAvg/ReadVariableOp?'batch_normalization_1/AssignMovingAvg_1?6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_1/batchnorm/ReadVariableOp?2batch_normalization_1/batchnorm/mul/ReadVariableOp?%batch_normalization_2/AssignMovingAvg?4batch_normalization_2/AssignMovingAvg/ReadVariableOp?'batch_normalization_2/AssignMovingAvg_1?6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?.batch_normalization_2/batchnorm/ReadVariableOp?2batch_normalization_2/batchnorm/mul/ReadVariableOp?conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsinputs%max_pooling1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
max_pooling1d/Squeeze?
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indices?
 batch_normalization/moments/meanMeanmax_pooling1d/Squeeze:output:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2"
 batch_normalization/moments/mean?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*#
_output_shapes
:?2*
(batch_normalization/moments/StopGradient?
-batch_normalization/moments/SquaredDifferenceSquaredDifferencemax_pooling1d/Squeeze:output:01batch_normalization/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2/
-batch_normalization/moments/SquaredDifference?
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indices?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2&
$batch_normalization/moments/variance?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1?
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)batch_normalization/AssignMovingAvg/decay?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:?2)
'batch_normalization/AssignMovingAvg/sub?
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2)
'batch_normalization/AssignMovingAvg/mul?
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02%
#batch_normalization/AssignMovingAvg?
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization/AssignMovingAvg_1/decay?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization/AssignMovingAvg_1/sub?
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization/AssignMovingAvg_1/mul?
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization/AssignMovingAvg_1?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/Rsqrt?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Mulmax_pooling1d/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2%
#batch_normalization/batchnorm/mul_1?
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/mul_2?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp?
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2%
#batch_normalization/batchnorm/add_1?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDims'batch_normalization/batchnorm/add_1:z:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2
conv1d/BiasAdd{
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2
conv1d/Relu?
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim?
max_pooling1d_1/ExpandDims
ExpandDimsconv1d/Relu:activations:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
max_pooling1d_1/ExpandDims?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPool?
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
max_pooling1d_1/Squeeze?
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indices?
"batch_normalization_1/moments/meanMean max_pooling1d_1/Squeeze:output:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2$
"batch_normalization_1/moments/mean?
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:?2,
*batch_normalization_1/moments/StopGradient?
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference max_pooling1d_1/Squeeze:output:03batch_normalization_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????21
/batch_normalization_1/moments/SquaredDifference?
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indices?
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2(
&batch_normalization_1/moments/variance?
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze?
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1?
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_1/AssignMovingAvg/decay?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp?
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_1/AssignMovingAvg/sub?
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2+
)batch_normalization_1/AssignMovingAvg/mul?
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_1/AssignMovingAvg?
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_1/AssignMovingAvg_1/decay?
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_1/AssignMovingAvg_1/sub?
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2-
+batch_normalization_1/AssignMovingAvg_1/mul?
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_1/AssignMovingAvg_1?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/Rsqrt?
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Mul max_pooling1d_1/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2'
%batch_normalization_1/batchnorm/mul_1?
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/mul_2?
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp?
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2'
%batch_normalization_1/batchnorm/add_1?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDims)batch_normalization_1/batchnorm/add_1:z:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?@*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2
conv1d_1/BiasAdd?
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_2/ExpandDims/dim?
max_pooling1d_2/ExpandDims
ExpandDimsconv1d_1/BiasAdd:output:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
max_pooling1d_2/ExpandDims?
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*8
_output_shapes&
$:"??????????????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d_2/MaxPool?
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2
max_pooling1d_2/Squeeze?
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indices?
"batch_normalization_2/moments/meanMean max_pooling1d_2/Squeeze:output:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2$
"batch_normalization_2/moments/mean?
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_2/moments/StopGradient?
/batch_normalization_2/moments/SquaredDifferenceSquaredDifference max_pooling1d_2/Squeeze:output:03batch_normalization_2/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????@21
/batch_normalization_2/moments/SquaredDifference?
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indices?
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2(
&batch_normalization_2/moments/variance?
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze?
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1?
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization_2/AssignMovingAvg/decay?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp?
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/sub?
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mul?
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02'
%batch_normalization_2/AssignMovingAvg?
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-batch_normalization_2/AssignMovingAvg_1/decay?
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/sub?
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mul?
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02)
'batch_normalization_2/AssignMovingAvg_1?
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_2/batchnorm/add/y?
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/add?
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrt?
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul?
%batch_normalization_2/batchnorm/mul_1Mul max_pooling1d_2/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2'
%batch_normalization_2/batchnorm/mul_1?
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2?
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp?
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/sub?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2'
%batch_normalization_2/batchnorm/add_1?
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:???????????????????: : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?+
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_91432

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_91474

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_2_layer_call_fn_91508

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_895632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_89734

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*8
_output_shapes&
$:"??????????????????@*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2	
Squeezeq
IdentityIdentitySqueeze:output:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????@:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?+
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_91562

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????@2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_90484
ipt
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?!
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?!

unknown_10:?@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16:@ 

unknown_17: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalliptunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_889922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
4
_output_shapes"
 :??????????????????

_user_specified_nameipt
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_91221

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2	
Squeezer
IdentityIdentitySqueeze:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????????????:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?	
?
3__inference_batch_normalization_layer_call_fn_91234

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_891232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_89123

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?+
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_89563

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????@2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
??
?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90676

inputsR
;embed_seq_1_el1_conv1d_expanddims_1_readvariableop_resource:?P
Aembed_seq_2_batch_normalization_batchnorm_readvariableop_resource:	?T
Eembed_seq_2_batch_normalization_batchnorm_mul_readvariableop_resource:	?R
Cembed_seq_2_batch_normalization_batchnorm_readvariableop_1_resource:	?R
Cembed_seq_2_batch_normalization_batchnorm_readvariableop_2_resource:	?V
>embed_seq_2_conv1d_conv1d_expanddims_1_readvariableop_resource:??A
2embed_seq_2_conv1d_biasadd_readvariableop_resource:	?R
Cembed_seq_2_batch_normalization_1_batchnorm_readvariableop_resource:	?V
Gembed_seq_2_batch_normalization_1_batchnorm_mul_readvariableop_resource:	?T
Eembed_seq_2_batch_normalization_1_batchnorm_readvariableop_1_resource:	?T
Eembed_seq_2_batch_normalization_1_batchnorm_readvariableop_2_resource:	?W
@embed_seq_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:?@B
4embed_seq_2_conv1d_1_biasadd_readvariableop_resource:@Q
Cembed_seq_2_batch_normalization_2_batchnorm_readvariableop_resource:@U
Gembed_seq_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:@S
Eembed_seq_2_batch_normalization_2_batchnorm_readvariableop_1_resource:@S
Eembed_seq_2_batch_normalization_2_batchnorm_readvariableop_2_resource:@J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:@ 6
(conv1d_2_biasadd_readvariableop_resource: 
identity??conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?2embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp?8embed_seq_2/batch_normalization/batchnorm/ReadVariableOp?:embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1?:embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2?<embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp?:embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp?<embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1?<embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2?>embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp?:embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp?<embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1?<embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2?>embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp?)embed_seq_2/conv1d/BiasAdd/ReadVariableOp?5embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp?+embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp?7embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
%embed_seq_1/el1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%embed_seq_1/el1/conv1d/ExpandDims/dim?
!embed_seq_1/el1/conv1d/ExpandDims
ExpandDimsinputs.embed_seq_1/el1/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2#
!embed_seq_1/el1/conv1d/ExpandDims?
2embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;embed_seq_1_el1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype024
2embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp?
'embed_seq_1/el1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'embed_seq_1/el1/conv1d/ExpandDims_1/dim?
#embed_seq_1/el1/conv1d/ExpandDims_1
ExpandDims:embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp:value:00embed_seq_1/el1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2%
#embed_seq_1/el1/conv1d/ExpandDims_1?
embed_seq_1/el1/conv1dConv2D*embed_seq_1/el1/conv1d/ExpandDims:output:0,embed_seq_1/el1/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
embed_seq_1/el1/conv1d?
embed_seq_1/el1/conv1d/SqueezeSqueezeembed_seq_1/el1/conv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2 
embed_seq_1/el1/conv1d/Squeeze?
(embed_seq_2/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(embed_seq_2/max_pooling1d/ExpandDims/dim?
$embed_seq_2/max_pooling1d/ExpandDims
ExpandDims'embed_seq_1/el1/conv1d/Squeeze:output:01embed_seq_2/max_pooling1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2&
$embed_seq_2/max_pooling1d/ExpandDims?
!embed_seq_2/max_pooling1d/MaxPoolMaxPool-embed_seq_2/max_pooling1d/ExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2#
!embed_seq_2/max_pooling1d/MaxPool?
!embed_seq_2/max_pooling1d/SqueezeSqueeze*embed_seq_2/max_pooling1d/MaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2#
!embed_seq_2/max_pooling1d/Squeeze?
8embed_seq_2/batch_normalization/batchnorm/ReadVariableOpReadVariableOpAembed_seq_2_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8embed_seq_2/batch_normalization/batchnorm/ReadVariableOp?
/embed_seq_2/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:21
/embed_seq_2/batch_normalization/batchnorm/add/y?
-embed_seq_2/batch_normalization/batchnorm/addAddV2@embed_seq_2/batch_normalization/batchnorm/ReadVariableOp:value:08embed_seq_2/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2/
-embed_seq_2/batch_normalization/batchnorm/add?
/embed_seq_2/batch_normalization/batchnorm/RsqrtRsqrt1embed_seq_2/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?21
/embed_seq_2/batch_normalization/batchnorm/Rsqrt?
<embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpEembed_seq_2_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp?
-embed_seq_2/batch_normalization/batchnorm/mulMul3embed_seq_2/batch_normalization/batchnorm/Rsqrt:y:0Dembed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2/
-embed_seq_2/batch_normalization/batchnorm/mul?
/embed_seq_2/batch_normalization/batchnorm/mul_1Mul*embed_seq_2/max_pooling1d/Squeeze:output:01embed_seq_2/batch_normalization/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????21
/embed_seq_2/batch_normalization/batchnorm/mul_1?
:embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpCembed_seq_2_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02<
:embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1?
/embed_seq_2/batch_normalization/batchnorm/mul_2MulBembed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1:value:01embed_seq_2/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?21
/embed_seq_2/batch_normalization/batchnorm/mul_2?
:embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpCembed_seq_2_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02<
:embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2?
-embed_seq_2/batch_normalization/batchnorm/subSubBembed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2:value:03embed_seq_2/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2/
-embed_seq_2/batch_normalization/batchnorm/sub?
/embed_seq_2/batch_normalization/batchnorm/add_1AddV23embed_seq_2/batch_normalization/batchnorm/mul_1:z:01embed_seq_2/batch_normalization/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????21
/embed_seq_2/batch_normalization/batchnorm/add_1?
(embed_seq_2/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(embed_seq_2/conv1d/conv1d/ExpandDims/dim?
$embed_seq_2/conv1d/conv1d/ExpandDims
ExpandDims3embed_seq_2/batch_normalization/batchnorm/add_1:z:01embed_seq_2/conv1d/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2&
$embed_seq_2/conv1d/conv1d/ExpandDims?
5embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>embed_seq_2_conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype027
5embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
*embed_seq_2/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*embed_seq_2/conv1d/conv1d/ExpandDims_1/dim?
&embed_seq_2/conv1d/conv1d/ExpandDims_1
ExpandDims=embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:03embed_seq_2/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2(
&embed_seq_2/conv1d/conv1d/ExpandDims_1?
embed_seq_2/conv1d/conv1dConv2D-embed_seq_2/conv1d/conv1d/ExpandDims:output:0/embed_seq_2/conv1d/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
embed_seq_2/conv1d/conv1d?
!embed_seq_2/conv1d/conv1d/SqueezeSqueeze"embed_seq_2/conv1d/conv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2#
!embed_seq_2/conv1d/conv1d/Squeeze?
)embed_seq_2/conv1d/BiasAdd/ReadVariableOpReadVariableOp2embed_seq_2_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)embed_seq_2/conv1d/BiasAdd/ReadVariableOp?
embed_seq_2/conv1d/BiasAddBiasAdd*embed_seq_2/conv1d/conv1d/Squeeze:output:01embed_seq_2/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2
embed_seq_2/conv1d/BiasAdd?
embed_seq_2/conv1d/ReluRelu#embed_seq_2/conv1d/BiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2
embed_seq_2/conv1d/Relu?
*embed_seq_2/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*embed_seq_2/max_pooling1d_1/ExpandDims/dim?
&embed_seq_2/max_pooling1d_1/ExpandDims
ExpandDims%embed_seq_2/conv1d/Relu:activations:03embed_seq_2/max_pooling1d_1/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2(
&embed_seq_2/max_pooling1d_1/ExpandDims?
#embed_seq_2/max_pooling1d_1/MaxPoolMaxPool/embed_seq_2/max_pooling1d_1/ExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2%
#embed_seq_2/max_pooling1d_1/MaxPool?
#embed_seq_2/max_pooling1d_1/SqueezeSqueeze,embed_seq_2/max_pooling1d_1/MaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2%
#embed_seq_2/max_pooling1d_1/Squeeze?
:embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpCembed_seq_2_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp?
1embed_seq_2/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1embed_seq_2/batch_normalization_1/batchnorm/add/y?
/embed_seq_2/batch_normalization_1/batchnorm/addAddV2Bembed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp:value:0:embed_seq_2/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?21
/embed_seq_2/batch_normalization_1/batchnorm/add?
1embed_seq_2/batch_normalization_1/batchnorm/RsqrtRsqrt3embed_seq_2/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?23
1embed_seq_2/batch_normalization_1/batchnorm/Rsqrt?
>embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpGembed_seq_2_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp?
/embed_seq_2/batch_normalization_1/batchnorm/mulMul5embed_seq_2/batch_normalization_1/batchnorm/Rsqrt:y:0Fembed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?21
/embed_seq_2/batch_normalization_1/batchnorm/mul?
1embed_seq_2/batch_normalization_1/batchnorm/mul_1Mul,embed_seq_2/max_pooling1d_1/Squeeze:output:03embed_seq_2/batch_normalization_1/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????23
1embed_seq_2/batch_normalization_1/batchnorm/mul_1?
<embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpEembed_seq_2_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02>
<embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1?
1embed_seq_2/batch_normalization_1/batchnorm/mul_2MulDembed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1:value:03embed_seq_2/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?23
1embed_seq_2/batch_normalization_1/batchnorm/mul_2?
<embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpEembed_seq_2_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02>
<embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2?
/embed_seq_2/batch_normalization_1/batchnorm/subSubDembed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2:value:05embed_seq_2/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?21
/embed_seq_2/batch_normalization_1/batchnorm/sub?
1embed_seq_2/batch_normalization_1/batchnorm/add_1AddV25embed_seq_2/batch_normalization_1/batchnorm/mul_1:z:03embed_seq_2/batch_normalization_1/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????23
1embed_seq_2/batch_normalization_1/batchnorm/add_1?
*embed_seq_2/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*embed_seq_2/conv1d_1/conv1d/ExpandDims/dim?
&embed_seq_2/conv1d_1/conv1d/ExpandDims
ExpandDims5embed_seq_2/batch_normalization_1/batchnorm/add_1:z:03embed_seq_2/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2(
&embed_seq_2/conv1d_1/conv1d/ExpandDims?
7embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@embed_seq_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?@*
dtype029
7embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
,embed_seq_2/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,embed_seq_2/conv1d_1/conv1d/ExpandDims_1/dim?
(embed_seq_2/conv1d_1/conv1d/ExpandDims_1
ExpandDims?embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:05embed_seq_2/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2*
(embed_seq_2/conv1d_1/conv1d/ExpandDims_1?
embed_seq_2/conv1d_1/conv1dConv2D/embed_seq_2/conv1d_1/conv1d/ExpandDims:output:01embed_seq_2/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
embed_seq_2/conv1d_1/conv1d?
#embed_seq_2/conv1d_1/conv1d/SqueezeSqueeze$embed_seq_2/conv1d_1/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims

?????????2%
#embed_seq_2/conv1d_1/conv1d/Squeeze?
+embed_seq_2/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp4embed_seq_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp?
embed_seq_2/conv1d_1/BiasAddBiasAdd,embed_seq_2/conv1d_1/conv1d/Squeeze:output:03embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2
embed_seq_2/conv1d_1/BiasAdd?
*embed_seq_2/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*embed_seq_2/max_pooling1d_2/ExpandDims/dim?
&embed_seq_2/max_pooling1d_2/ExpandDims
ExpandDims%embed_seq_2/conv1d_1/BiasAdd:output:03embed_seq_2/max_pooling1d_2/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2(
&embed_seq_2/max_pooling1d_2/ExpandDims?
#embed_seq_2/max_pooling1d_2/MaxPoolMaxPool/embed_seq_2/max_pooling1d_2/ExpandDims:output:0*8
_output_shapes&
$:"??????????????????@*
ksize
*
paddingVALID*
strides
2%
#embed_seq_2/max_pooling1d_2/MaxPool?
#embed_seq_2/max_pooling1d_2/SqueezeSqueeze,embed_seq_2/max_pooling1d_2/MaxPool:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2%
#embed_seq_2/max_pooling1d_2/Squeeze?
:embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpCembed_seq_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp?
1embed_seq_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1embed_seq_2/batch_normalization_2/batchnorm/add/y?
/embed_seq_2/batch_normalization_2/batchnorm/addAddV2Bembed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0:embed_seq_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@21
/embed_seq_2/batch_normalization_2/batchnorm/add?
1embed_seq_2/batch_normalization_2/batchnorm/RsqrtRsqrt3embed_seq_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@23
1embed_seq_2/batch_normalization_2/batchnorm/Rsqrt?
>embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpGembed_seq_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp?
/embed_seq_2/batch_normalization_2/batchnorm/mulMul5embed_seq_2/batch_normalization_2/batchnorm/Rsqrt:y:0Fembed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/embed_seq_2/batch_normalization_2/batchnorm/mul?
1embed_seq_2/batch_normalization_2/batchnorm/mul_1Mul,embed_seq_2/max_pooling1d_2/Squeeze:output:03embed_seq_2/batch_normalization_2/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@23
1embed_seq_2/batch_normalization_2/batchnorm/mul_1?
<embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpEembed_seq_2_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02>
<embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1?
1embed_seq_2/batch_normalization_2/batchnorm/mul_2MulDembed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1:value:03embed_seq_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@23
1embed_seq_2/batch_normalization_2/batchnorm/mul_2?
<embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpEembed_seq_2_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02>
<embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2?
/embed_seq_2/batch_normalization_2/batchnorm/subSubDembed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2:value:05embed_seq_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@21
/embed_seq_2/batch_normalization_2/batchnorm/sub?
1embed_seq_2/batch_normalization_2/batchnorm/add_1AddV25embed_seq_2/batch_normalization_2/batchnorm/mul_1:z:03embed_seq_2/batch_normalization_2/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@23
1embed_seq_2/batch_normalization_2/batchnorm/add_1?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDims5embed_seq_2/batch_normalization_2/batchnorm/add_1:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? *
paddingSAME*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*4
_output_shapes"
 :?????????????????? *
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
conv1d_2/BiasAdd?
IdentityIdentityconv1d_2/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?	
NoOpNoOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp3^embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp9^embed_seq_2/batch_normalization/batchnorm/ReadVariableOp;^embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1;^embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2=^embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp;^embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp=^embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1=^embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2?^embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp;^embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp=^embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1=^embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2?^embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*^embed_seq_2/conv1d/BiasAdd/ReadVariableOp6^embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp,^embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp8^embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2h
2embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp2embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp2t
8embed_seq_2/batch_normalization/batchnorm/ReadVariableOp8embed_seq_2/batch_normalization/batchnorm/ReadVariableOp2x
:embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1:embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_12x
:embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2:embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_22|
<embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp<embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp2x
:embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp:embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp2|
<embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1<embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_12|
<embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2<embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_22?
>embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp>embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp2x
:embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp:embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp2|
<embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1<embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_12|
<embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2<embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_22?
>embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp>embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp2V
)embed_seq_2/conv1d/BiasAdd/ReadVariableOp)embed_seq_2/conv1d/BiasAdd/ReadVariableOp2n
5embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp5embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp2Z
+embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp+embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp2r
7embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp7embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
)__inference_embed_seq_layer_call_fn_90349
ipt
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?!
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?!

unknown_10:?@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16:@ 

unknown_17: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalliptunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? */
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_embed_seq_layer_call_and_return_conditional_losses_902652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
4
_output_shapes"
 :??????????????????

_user_specified_nameipt
?
?
+__inference_embed_seq_1_layer_call_fn_89021
	el1_input
unknown:?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	el1_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_890162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
4
_output_shapes"
 :??????????????????
#
_user_specified_name	el1_input
?
?
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_90862

inputsF
/el1_conv1d_expanddims_1_readvariableop_resource:?
identity??&el1/conv1d/ExpandDims_1/ReadVariableOp?
el1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
el1/conv1d/ExpandDims/dim?
el1/conv1d/ExpandDims
ExpandDimsinputs"el1/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
el1/conv1d/ExpandDims?
&el1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp/el1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02(
&el1/conv1d/ExpandDims_1/ReadVariableOp|
el1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
el1/conv1d/ExpandDims_1/dim?
el1/conv1d/ExpandDims_1
ExpandDims.el1/conv1d/ExpandDims_1/ReadVariableOp:value:0$el1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2
el1/conv1d/ExpandDims_1?

el1/conv1dConv2Del1/conv1d/ExpandDims:output:0 el1/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2

el1/conv1d?
el1/conv1d/SqueezeSqueezeel1/conv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2
el1/conv1d/Squeeze?
IdentityIdentityel1/conv1d/Squeeze:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityw
NoOpNoOp'^el1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 2P
&el1/conv1d/ExpandDims_1/ReadVariableOp&el1/conv1d/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_89655

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2	
Squeezer
IdentityIdentitySqueeze:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????????????:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
+__inference_embed_seq_1_layer_call_fn_89057
	el1_input
unknown:?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	el1_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_890452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
4
_output_shapes"
 :??????????????????
#
_user_specified_name	el1_input
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_89083

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?+
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_89183

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv1d_1_layer_call_fn_91441

inputs
unknown:?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_897212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?+
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_91301

inputs6
'assignmovingavg_readvariableop_resource:	?8
)assignmovingavg_1_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?0
!batchnorm_readvariableop_resource:	?
identity??AssignMovingAvg?AssignMovingAvg/ReadVariableOp?AssignMovingAvg_1? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1s
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg/mul?
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvgw
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignMovingAvg_1g
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
+__inference_embed_seq_2_layer_call_fn_89781
max_pooling1d_input
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	? 
	unknown_9:?@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallmax_pooling1d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_897462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:???????????????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
5
_output_shapes#
!:???????????????????
-
_user_specified_namemax_pooling1d_input
?T
?
!__inference__traced_restore_91709
file_prefix6
 assignvariableop_conv1d_2_kernel:@ .
 assignvariableop_1_conv1d_2_bias: ;
,assignvariableop_2_batch_normalization_gamma:	?:
+assignvariableop_3_batch_normalization_beta:	?8
 assignvariableop_4_conv1d_kernel:??-
assignvariableop_5_conv1d_bias:	?=
.assignvariableop_6_batch_normalization_1_gamma:	?<
-assignvariableop_7_batch_normalization_1_beta:	?9
"assignvariableop_8_conv1d_1_kernel:?@.
 assignvariableop_9_conv1d_1_bias:@=
/assignvariableop_10_batch_normalization_2_gamma:@<
.assignvariableop_11_batch_normalization_2_beta:@5
assignvariableop_12_el1_kernel:?B
3assignvariableop_13_batch_normalization_moving_mean:	?F
7assignvariableop_14_batch_normalization_moving_variance:	?D
5assignvariableop_15_batch_normalization_1_moving_mean:	?H
9assignvariableop_16_batch_normalization_1_moving_variance:	?C
5assignvariableop_17_batch_normalization_2_moving_mean:@G
9assignvariableop_18_batch_normalization_2_moving_variance:@
identity_20??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv1d_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv1d_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv1d_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv1d_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv1d_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv1d_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_el1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp3assignvariableop_13_batch_normalization_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp7assignvariableop_14_batch_normalization_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp5assignvariableop_15_batch_normalization_1_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp9assignvariableop_16_batch_normalization_1_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp5assignvariableop_17_batch_normalization_2_moving_meanIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp9assignvariableop_18_batch_normalization_2_moving_varianceIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19f
Identity_20IdentityIdentity_19:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_20?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_20Identity_20:output:0*;
_input_shapes*
(: : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
I
-__inference_max_pooling1d_layer_call_fn_91200

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_890832
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?	
?
3__inference_batch_normalization_layer_call_fn_91247

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_891832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?.
?
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_90015
max_pooling1d_input(
batch_normalization_89975:	?(
batch_normalization_89977:	?(
batch_normalization_89979:	?(
batch_normalization_89981:	?$
conv1d_89984:??
conv1d_89986:	?*
batch_normalization_1_89990:	?*
batch_normalization_1_89992:	?*
batch_normalization_1_89994:	?*
batch_normalization_1_89996:	?%
conv1d_1_89999:?@
conv1d_1_90001:@)
batch_normalization_2_90005:@)
batch_normalization_2_90007:@)
batch_normalization_2_90009:@)
batch_normalization_2_90011:@
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCallmax_pooling1d_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_896552
max_pooling1d/PartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0batch_normalization_89975batch_normalization_89977batch_normalization_89979batch_normalization_89981*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_891232-
+batch_normalization/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv1d_89984conv1d_89986*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_896822 
conv1d/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_896952!
max_pooling1d_1/PartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0batch_normalization_1_89990batch_normalization_1_89992batch_normalization_1_89994batch_normalization_1_89996*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_893132/
-batch_normalization_1/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv1d_1_89999conv1d_1_90001*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_897212"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_897342!
max_pooling1d_2/PartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0batch_normalization_2_90005batch_normalization_2_90007batch_normalization_2_90009batch_normalization_2_90011*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_895032/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:???????????????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall:j f
5
_output_shapes#
!:???????????????????
-
_user_specified_namemax_pooling1d_input
?	
?
5__inference_batch_normalization_1_layer_call_fn_91378

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_893732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
??
?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90824

inputsR
;embed_seq_1_el1_conv1d_expanddims_1_readvariableop_resource:?V
Gembed_seq_2_batch_normalization_assignmovingavg_readvariableop_resource:	?X
Iembed_seq_2_batch_normalization_assignmovingavg_1_readvariableop_resource:	?T
Eembed_seq_2_batch_normalization_batchnorm_mul_readvariableop_resource:	?P
Aembed_seq_2_batch_normalization_batchnorm_readvariableop_resource:	?V
>embed_seq_2_conv1d_conv1d_expanddims_1_readvariableop_resource:??A
2embed_seq_2_conv1d_biasadd_readvariableop_resource:	?X
Iembed_seq_2_batch_normalization_1_assignmovingavg_readvariableop_resource:	?Z
Kembed_seq_2_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	?V
Gembed_seq_2_batch_normalization_1_batchnorm_mul_readvariableop_resource:	?R
Cembed_seq_2_batch_normalization_1_batchnorm_readvariableop_resource:	?W
@embed_seq_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:?@B
4embed_seq_2_conv1d_1_biasadd_readvariableop_resource:@W
Iembed_seq_2_batch_normalization_2_assignmovingavg_readvariableop_resource:@Y
Kembed_seq_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource:@U
Gembed_seq_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:@Q
Cembed_seq_2_batch_normalization_2_batchnorm_readvariableop_resource:@J
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:@ 6
(conv1d_2_biasadd_readvariableop_resource: 
identity??conv1d_2/BiasAdd/ReadVariableOp?+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?2embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp?/embed_seq_2/batch_normalization/AssignMovingAvg?>embed_seq_2/batch_normalization/AssignMovingAvg/ReadVariableOp?1embed_seq_2/batch_normalization/AssignMovingAvg_1?@embed_seq_2/batch_normalization/AssignMovingAvg_1/ReadVariableOp?8embed_seq_2/batch_normalization/batchnorm/ReadVariableOp?<embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp?1embed_seq_2/batch_normalization_1/AssignMovingAvg?@embed_seq_2/batch_normalization_1/AssignMovingAvg/ReadVariableOp?3embed_seq_2/batch_normalization_1/AssignMovingAvg_1?Bembed_seq_2/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?:embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp?>embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp?1embed_seq_2/batch_normalization_2/AssignMovingAvg?@embed_seq_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp?3embed_seq_2/batch_normalization_2/AssignMovingAvg_1?Bembed_seq_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?:embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp?>embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp?)embed_seq_2/conv1d/BiasAdd/ReadVariableOp?5embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp?+embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp?7embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
%embed_seq_1/el1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%embed_seq_1/el1/conv1d/ExpandDims/dim?
!embed_seq_1/el1/conv1d/ExpandDims
ExpandDimsinputs.embed_seq_1/el1/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2#
!embed_seq_1/el1/conv1d/ExpandDims?
2embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp;embed_seq_1_el1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype024
2embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp?
'embed_seq_1/el1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2)
'embed_seq_1/el1/conv1d/ExpandDims_1/dim?
#embed_seq_1/el1/conv1d/ExpandDims_1
ExpandDims:embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp:value:00embed_seq_1/el1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2%
#embed_seq_1/el1/conv1d/ExpandDims_1?
embed_seq_1/el1/conv1dConv2D*embed_seq_1/el1/conv1d/ExpandDims:output:0,embed_seq_1/el1/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
embed_seq_1/el1/conv1d?
embed_seq_1/el1/conv1d/SqueezeSqueezeembed_seq_1/el1/conv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2 
embed_seq_1/el1/conv1d/Squeeze?
(embed_seq_2/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(embed_seq_2/max_pooling1d/ExpandDims/dim?
$embed_seq_2/max_pooling1d/ExpandDims
ExpandDims'embed_seq_1/el1/conv1d/Squeeze:output:01embed_seq_2/max_pooling1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2&
$embed_seq_2/max_pooling1d/ExpandDims?
!embed_seq_2/max_pooling1d/MaxPoolMaxPool-embed_seq_2/max_pooling1d/ExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2#
!embed_seq_2/max_pooling1d/MaxPool?
!embed_seq_2/max_pooling1d/SqueezeSqueeze*embed_seq_2/max_pooling1d/MaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2#
!embed_seq_2/max_pooling1d/Squeeze?
>embed_seq_2/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2@
>embed_seq_2/batch_normalization/moments/mean/reduction_indices?
,embed_seq_2/batch_normalization/moments/meanMean*embed_seq_2/max_pooling1d/Squeeze:output:0Gembed_seq_2/batch_normalization/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2.
,embed_seq_2/batch_normalization/moments/mean?
4embed_seq_2/batch_normalization/moments/StopGradientStopGradient5embed_seq_2/batch_normalization/moments/mean:output:0*
T0*#
_output_shapes
:?26
4embed_seq_2/batch_normalization/moments/StopGradient?
9embed_seq_2/batch_normalization/moments/SquaredDifferenceSquaredDifference*embed_seq_2/max_pooling1d/Squeeze:output:0=embed_seq_2/batch_normalization/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2;
9embed_seq_2/batch_normalization/moments/SquaredDifference?
Bembed_seq_2/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2D
Bembed_seq_2/batch_normalization/moments/variance/reduction_indices?
0embed_seq_2/batch_normalization/moments/varianceMean=embed_seq_2/batch_normalization/moments/SquaredDifference:z:0Kembed_seq_2/batch_normalization/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(22
0embed_seq_2/batch_normalization/moments/variance?
/embed_seq_2/batch_normalization/moments/SqueezeSqueeze5embed_seq_2/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 21
/embed_seq_2/batch_normalization/moments/Squeeze?
1embed_seq_2/batch_normalization/moments/Squeeze_1Squeeze9embed_seq_2/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 23
1embed_seq_2/batch_normalization/moments/Squeeze_1?
5embed_seq_2/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<27
5embed_seq_2/batch_normalization/AssignMovingAvg/decay?
>embed_seq_2/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpGembed_seq_2_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>embed_seq_2/batch_normalization/AssignMovingAvg/ReadVariableOp?
3embed_seq_2/batch_normalization/AssignMovingAvg/subSubFembed_seq_2/batch_normalization/AssignMovingAvg/ReadVariableOp:value:08embed_seq_2/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:?25
3embed_seq_2/batch_normalization/AssignMovingAvg/sub?
3embed_seq_2/batch_normalization/AssignMovingAvg/mulMul7embed_seq_2/batch_normalization/AssignMovingAvg/sub:z:0>embed_seq_2/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?25
3embed_seq_2/batch_normalization/AssignMovingAvg/mul?
/embed_seq_2/batch_normalization/AssignMovingAvgAssignSubVariableOpGembed_seq_2_batch_normalization_assignmovingavg_readvariableop_resource7embed_seq_2/batch_normalization/AssignMovingAvg/mul:z:0?^embed_seq_2/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype021
/embed_seq_2/batch_normalization/AssignMovingAvg?
7embed_seq_2/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<29
7embed_seq_2/batch_normalization/AssignMovingAvg_1/decay?
@embed_seq_2/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpIembed_seq_2_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02B
@embed_seq_2/batch_normalization/AssignMovingAvg_1/ReadVariableOp?
5embed_seq_2/batch_normalization/AssignMovingAvg_1/subSubHembed_seq_2/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0:embed_seq_2/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?27
5embed_seq_2/batch_normalization/AssignMovingAvg_1/sub?
5embed_seq_2/batch_normalization/AssignMovingAvg_1/mulMul9embed_seq_2/batch_normalization/AssignMovingAvg_1/sub:z:0@embed_seq_2/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?27
5embed_seq_2/batch_normalization/AssignMovingAvg_1/mul?
1embed_seq_2/batch_normalization/AssignMovingAvg_1AssignSubVariableOpIembed_seq_2_batch_normalization_assignmovingavg_1_readvariableop_resource9embed_seq_2/batch_normalization/AssignMovingAvg_1/mul:z:0A^embed_seq_2/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype023
1embed_seq_2/batch_normalization/AssignMovingAvg_1?
/embed_seq_2/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:21
/embed_seq_2/batch_normalization/batchnorm/add/y?
-embed_seq_2/batch_normalization/batchnorm/addAddV2:embed_seq_2/batch_normalization/moments/Squeeze_1:output:08embed_seq_2/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2/
-embed_seq_2/batch_normalization/batchnorm/add?
/embed_seq_2/batch_normalization/batchnorm/RsqrtRsqrt1embed_seq_2/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?21
/embed_seq_2/batch_normalization/batchnorm/Rsqrt?
<embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpEembed_seq_2_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02>
<embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp?
-embed_seq_2/batch_normalization/batchnorm/mulMul3embed_seq_2/batch_normalization/batchnorm/Rsqrt:y:0Dembed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2/
-embed_seq_2/batch_normalization/batchnorm/mul?
/embed_seq_2/batch_normalization/batchnorm/mul_1Mul*embed_seq_2/max_pooling1d/Squeeze:output:01embed_seq_2/batch_normalization/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????21
/embed_seq_2/batch_normalization/batchnorm/mul_1?
/embed_seq_2/batch_normalization/batchnorm/mul_2Mul8embed_seq_2/batch_normalization/moments/Squeeze:output:01embed_seq_2/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?21
/embed_seq_2/batch_normalization/batchnorm/mul_2?
8embed_seq_2/batch_normalization/batchnorm/ReadVariableOpReadVariableOpAembed_seq_2_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8embed_seq_2/batch_normalization/batchnorm/ReadVariableOp?
-embed_seq_2/batch_normalization/batchnorm/subSub@embed_seq_2/batch_normalization/batchnorm/ReadVariableOp:value:03embed_seq_2/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2/
-embed_seq_2/batch_normalization/batchnorm/sub?
/embed_seq_2/batch_normalization/batchnorm/add_1AddV23embed_seq_2/batch_normalization/batchnorm/mul_1:z:01embed_seq_2/batch_normalization/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????21
/embed_seq_2/batch_normalization/batchnorm/add_1?
(embed_seq_2/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(embed_seq_2/conv1d/conv1d/ExpandDims/dim?
$embed_seq_2/conv1d/conv1d/ExpandDims
ExpandDims3embed_seq_2/batch_normalization/batchnorm/add_1:z:01embed_seq_2/conv1d/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2&
$embed_seq_2/conv1d/conv1d/ExpandDims?
5embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>embed_seq_2_conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype027
5embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
*embed_seq_2/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*embed_seq_2/conv1d/conv1d/ExpandDims_1/dim?
&embed_seq_2/conv1d/conv1d/ExpandDims_1
ExpandDims=embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:03embed_seq_2/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2(
&embed_seq_2/conv1d/conv1d/ExpandDims_1?
embed_seq_2/conv1d/conv1dConv2D-embed_seq_2/conv1d/conv1d/ExpandDims:output:0/embed_seq_2/conv1d/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
embed_seq_2/conv1d/conv1d?
!embed_seq_2/conv1d/conv1d/SqueezeSqueeze"embed_seq_2/conv1d/conv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2#
!embed_seq_2/conv1d/conv1d/Squeeze?
)embed_seq_2/conv1d/BiasAdd/ReadVariableOpReadVariableOp2embed_seq_2_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)embed_seq_2/conv1d/BiasAdd/ReadVariableOp?
embed_seq_2/conv1d/BiasAddBiasAdd*embed_seq_2/conv1d/conv1d/Squeeze:output:01embed_seq_2/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2
embed_seq_2/conv1d/BiasAdd?
embed_seq_2/conv1d/ReluRelu#embed_seq_2/conv1d/BiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2
embed_seq_2/conv1d/Relu?
*embed_seq_2/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*embed_seq_2/max_pooling1d_1/ExpandDims/dim?
&embed_seq_2/max_pooling1d_1/ExpandDims
ExpandDims%embed_seq_2/conv1d/Relu:activations:03embed_seq_2/max_pooling1d_1/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2(
&embed_seq_2/max_pooling1d_1/ExpandDims?
#embed_seq_2/max_pooling1d_1/MaxPoolMaxPool/embed_seq_2/max_pooling1d_1/ExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2%
#embed_seq_2/max_pooling1d_1/MaxPool?
#embed_seq_2/max_pooling1d_1/SqueezeSqueeze,embed_seq_2/max_pooling1d_1/MaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2%
#embed_seq_2/max_pooling1d_1/Squeeze?
@embed_seq_2/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@embed_seq_2/batch_normalization_1/moments/mean/reduction_indices?
.embed_seq_2/batch_normalization_1/moments/meanMean,embed_seq_2/max_pooling1d_1/Squeeze:output:0Iembed_seq_2/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(20
.embed_seq_2/batch_normalization_1/moments/mean?
6embed_seq_2/batch_normalization_1/moments/StopGradientStopGradient7embed_seq_2/batch_normalization_1/moments/mean:output:0*
T0*#
_output_shapes
:?28
6embed_seq_2/batch_normalization_1/moments/StopGradient?
;embed_seq_2/batch_normalization_1/moments/SquaredDifferenceSquaredDifference,embed_seq_2/max_pooling1d_1/Squeeze:output:0?embed_seq_2/batch_normalization_1/moments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2=
;embed_seq_2/batch_normalization_1/moments/SquaredDifference?
Dembed_seq_2/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dembed_seq_2/batch_normalization_1/moments/variance/reduction_indices?
2embed_seq_2/batch_normalization_1/moments/varianceMean?embed_seq_2/batch_normalization_1/moments/SquaredDifference:z:0Membed_seq_2/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(24
2embed_seq_2/batch_normalization_1/moments/variance?
1embed_seq_2/batch_normalization_1/moments/SqueezeSqueeze7embed_seq_2/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 23
1embed_seq_2/batch_normalization_1/moments/Squeeze?
3embed_seq_2/batch_normalization_1/moments/Squeeze_1Squeeze;embed_seq_2/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 25
3embed_seq_2/batch_normalization_1/moments/Squeeze_1?
7embed_seq_2/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<29
7embed_seq_2/batch_normalization_1/AssignMovingAvg/decay?
@embed_seq_2/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpIembed_seq_2_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:?*
dtype02B
@embed_seq_2/batch_normalization_1/AssignMovingAvg/ReadVariableOp?
5embed_seq_2/batch_normalization_1/AssignMovingAvg/subSubHembed_seq_2/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0:embed_seq_2/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:?27
5embed_seq_2/batch_normalization_1/AssignMovingAvg/sub?
5embed_seq_2/batch_normalization_1/AssignMovingAvg/mulMul9embed_seq_2/batch_normalization_1/AssignMovingAvg/sub:z:0@embed_seq_2/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:?27
5embed_seq_2/batch_normalization_1/AssignMovingAvg/mul?
1embed_seq_2/batch_normalization_1/AssignMovingAvgAssignSubVariableOpIembed_seq_2_batch_normalization_1_assignmovingavg_readvariableop_resource9embed_seq_2/batch_normalization_1/AssignMovingAvg/mul:z:0A^embed_seq_2/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype023
1embed_seq_2/batch_normalization_1/AssignMovingAvg?
9embed_seq_2/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2;
9embed_seq_2/batch_normalization_1/AssignMovingAvg_1/decay?
Bembed_seq_2/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpKembed_seq_2_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:?*
dtype02D
Bembed_seq_2/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp?
7embed_seq_2/batch_normalization_1/AssignMovingAvg_1/subSubJembed_seq_2/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:0<embed_seq_2/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:?29
7embed_seq_2/batch_normalization_1/AssignMovingAvg_1/sub?
7embed_seq_2/batch_normalization_1/AssignMovingAvg_1/mulMul;embed_seq_2/batch_normalization_1/AssignMovingAvg_1/sub:z:0Bembed_seq_2/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:?29
7embed_seq_2/batch_normalization_1/AssignMovingAvg_1/mul?
3embed_seq_2/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpKembed_seq_2_batch_normalization_1_assignmovingavg_1_readvariableop_resource;embed_seq_2/batch_normalization_1/AssignMovingAvg_1/mul:z:0C^embed_seq_2/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype025
3embed_seq_2/batch_normalization_1/AssignMovingAvg_1?
1embed_seq_2/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1embed_seq_2/batch_normalization_1/batchnorm/add/y?
/embed_seq_2/batch_normalization_1/batchnorm/addAddV2<embed_seq_2/batch_normalization_1/moments/Squeeze_1:output:0:embed_seq_2/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?21
/embed_seq_2/batch_normalization_1/batchnorm/add?
1embed_seq_2/batch_normalization_1/batchnorm/RsqrtRsqrt3embed_seq_2/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?23
1embed_seq_2/batch_normalization_1/batchnorm/Rsqrt?
>embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpGembed_seq_2_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02@
>embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp?
/embed_seq_2/batch_normalization_1/batchnorm/mulMul5embed_seq_2/batch_normalization_1/batchnorm/Rsqrt:y:0Fembed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?21
/embed_seq_2/batch_normalization_1/batchnorm/mul?
1embed_seq_2/batch_normalization_1/batchnorm/mul_1Mul,embed_seq_2/max_pooling1d_1/Squeeze:output:03embed_seq_2/batch_normalization_1/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????23
1embed_seq_2/batch_normalization_1/batchnorm/mul_1?
1embed_seq_2/batch_normalization_1/batchnorm/mul_2Mul:embed_seq_2/batch_normalization_1/moments/Squeeze:output:03embed_seq_2/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?23
1embed_seq_2/batch_normalization_1/batchnorm/mul_2?
:embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpCembed_seq_2_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02<
:embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp?
/embed_seq_2/batch_normalization_1/batchnorm/subSubBembed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp:value:05embed_seq_2/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?21
/embed_seq_2/batch_normalization_1/batchnorm/sub?
1embed_seq_2/batch_normalization_1/batchnorm/add_1AddV25embed_seq_2/batch_normalization_1/batchnorm/mul_1:z:03embed_seq_2/batch_normalization_1/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????23
1embed_seq_2/batch_normalization_1/batchnorm/add_1?
*embed_seq_2/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*embed_seq_2/conv1d_1/conv1d/ExpandDims/dim?
&embed_seq_2/conv1d_1/conv1d/ExpandDims
ExpandDims5embed_seq_2/batch_normalization_1/batchnorm/add_1:z:03embed_seq_2/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2(
&embed_seq_2/conv1d_1/conv1d/ExpandDims?
7embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@embed_seq_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?@*
dtype029
7embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
,embed_seq_2/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,embed_seq_2/conv1d_1/conv1d/ExpandDims_1/dim?
(embed_seq_2/conv1d_1/conv1d/ExpandDims_1
ExpandDims?embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:05embed_seq_2/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2*
(embed_seq_2/conv1d_1/conv1d/ExpandDims_1?
embed_seq_2/conv1d_1/conv1dConv2D/embed_seq_2/conv1d_1/conv1d/ExpandDims:output:01embed_seq_2/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
embed_seq_2/conv1d_1/conv1d?
#embed_seq_2/conv1d_1/conv1d/SqueezeSqueeze$embed_seq_2/conv1d_1/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims

?????????2%
#embed_seq_2/conv1d_1/conv1d/Squeeze?
+embed_seq_2/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp4embed_seq_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp?
embed_seq_2/conv1d_1/BiasAddBiasAdd,embed_seq_2/conv1d_1/conv1d/Squeeze:output:03embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2
embed_seq_2/conv1d_1/BiasAdd?
*embed_seq_2/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*embed_seq_2/max_pooling1d_2/ExpandDims/dim?
&embed_seq_2/max_pooling1d_2/ExpandDims
ExpandDims%embed_seq_2/conv1d_1/BiasAdd:output:03embed_seq_2/max_pooling1d_2/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2(
&embed_seq_2/max_pooling1d_2/ExpandDims?
#embed_seq_2/max_pooling1d_2/MaxPoolMaxPool/embed_seq_2/max_pooling1d_2/ExpandDims:output:0*8
_output_shapes&
$:"??????????????????@*
ksize
*
paddingVALID*
strides
2%
#embed_seq_2/max_pooling1d_2/MaxPool?
#embed_seq_2/max_pooling1d_2/SqueezeSqueeze,embed_seq_2/max_pooling1d_2/MaxPool:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2%
#embed_seq_2/max_pooling1d_2/Squeeze?
@embed_seq_2/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2B
@embed_seq_2/batch_normalization_2/moments/mean/reduction_indices?
.embed_seq_2/batch_normalization_2/moments/meanMean,embed_seq_2/max_pooling1d_2/Squeeze:output:0Iembed_seq_2/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(20
.embed_seq_2/batch_normalization_2/moments/mean?
6embed_seq_2/batch_normalization_2/moments/StopGradientStopGradient7embed_seq_2/batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@28
6embed_seq_2/batch_normalization_2/moments/StopGradient?
;embed_seq_2/batch_normalization_2/moments/SquaredDifferenceSquaredDifference,embed_seq_2/max_pooling1d_2/Squeeze:output:0?embed_seq_2/batch_normalization_2/moments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????@2=
;embed_seq_2/batch_normalization_2/moments/SquaredDifference?
Dembed_seq_2/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2F
Dembed_seq_2/batch_normalization_2/moments/variance/reduction_indices?
2embed_seq_2/batch_normalization_2/moments/varianceMean?embed_seq_2/batch_normalization_2/moments/SquaredDifference:z:0Membed_seq_2/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(24
2embed_seq_2/batch_normalization_2/moments/variance?
1embed_seq_2/batch_normalization_2/moments/SqueezeSqueeze7embed_seq_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 23
1embed_seq_2/batch_normalization_2/moments/Squeeze?
3embed_seq_2/batch_normalization_2/moments/Squeeze_1Squeeze;embed_seq_2/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 25
3embed_seq_2/batch_normalization_2/moments/Squeeze_1?
7embed_seq_2/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<29
7embed_seq_2/batch_normalization_2/AssignMovingAvg/decay?
@embed_seq_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpIembed_seq_2_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype02B
@embed_seq_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp?
5embed_seq_2/batch_normalization_2/AssignMovingAvg/subSubHembed_seq_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0:embed_seq_2/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:@27
5embed_seq_2/batch_normalization_2/AssignMovingAvg/sub?
5embed_seq_2/batch_normalization_2/AssignMovingAvg/mulMul9embed_seq_2/batch_normalization_2/AssignMovingAvg/sub:z:0@embed_seq_2/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@27
5embed_seq_2/batch_normalization_2/AssignMovingAvg/mul?
1embed_seq_2/batch_normalization_2/AssignMovingAvgAssignSubVariableOpIembed_seq_2_batch_normalization_2_assignmovingavg_readvariableop_resource9embed_seq_2/batch_normalization_2/AssignMovingAvg/mul:z:0A^embed_seq_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype023
1embed_seq_2/batch_normalization_2/AssignMovingAvg?
9embed_seq_2/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2;
9embed_seq_2/batch_normalization_2/AssignMovingAvg_1/decay?
Bembed_seq_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpKembed_seq_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bembed_seq_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp?
7embed_seq_2/batch_normalization_2/AssignMovingAvg_1/subSubJembed_seq_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0<embed_seq_2/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@29
7embed_seq_2/batch_normalization_2/AssignMovingAvg_1/sub?
7embed_seq_2/batch_normalization_2/AssignMovingAvg_1/mulMul;embed_seq_2/batch_normalization_2/AssignMovingAvg_1/sub:z:0Bembed_seq_2/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@29
7embed_seq_2/batch_normalization_2/AssignMovingAvg_1/mul?
3embed_seq_2/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpKembed_seq_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource;embed_seq_2/batch_normalization_2/AssignMovingAvg_1/mul:z:0C^embed_seq_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype025
3embed_seq_2/batch_normalization_2/AssignMovingAvg_1?
1embed_seq_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1embed_seq_2/batch_normalization_2/batchnorm/add/y?
/embed_seq_2/batch_normalization_2/batchnorm/addAddV2<embed_seq_2/batch_normalization_2/moments/Squeeze_1:output:0:embed_seq_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@21
/embed_seq_2/batch_normalization_2/batchnorm/add?
1embed_seq_2/batch_normalization_2/batchnorm/RsqrtRsqrt3embed_seq_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@23
1embed_seq_2/batch_normalization_2/batchnorm/Rsqrt?
>embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpGembed_seq_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02@
>embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp?
/embed_seq_2/batch_normalization_2/batchnorm/mulMul5embed_seq_2/batch_normalization_2/batchnorm/Rsqrt:y:0Fembed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@21
/embed_seq_2/batch_normalization_2/batchnorm/mul?
1embed_seq_2/batch_normalization_2/batchnorm/mul_1Mul,embed_seq_2/max_pooling1d_2/Squeeze:output:03embed_seq_2/batch_normalization_2/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@23
1embed_seq_2/batch_normalization_2/batchnorm/mul_1?
1embed_seq_2/batch_normalization_2/batchnorm/mul_2Mul:embed_seq_2/batch_normalization_2/moments/Squeeze:output:03embed_seq_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@23
1embed_seq_2/batch_normalization_2/batchnorm/mul_2?
:embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpCembed_seq_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02<
:embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp?
/embed_seq_2/batch_normalization_2/batchnorm/subSubBembed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp:value:05embed_seq_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@21
/embed_seq_2/batch_normalization_2/batchnorm/sub?
1embed_seq_2/batch_normalization_2/batchnorm/add_1AddV25embed_seq_2/batch_normalization_2/batchnorm/mul_1:z:03embed_seq_2/batch_normalization_2/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@23
1embed_seq_2/batch_normalization_2/batchnorm/add_1?
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_2/conv1d/ExpandDims/dim?
conv1d_2/conv1d/ExpandDims
ExpandDims5embed_seq_2/batch_normalization_2/batchnorm/add_1:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d_2/conv1d/ExpandDims?
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim?
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d_2/conv1d/ExpandDims_1?
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? *
paddingSAME*
strides
2
conv1d_2/conv1d?
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*4
_output_shapes"
 :?????????????????? *
squeeze_dims

?????????2
conv1d_2/conv1d/Squeeze?
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_2/BiasAdd/ReadVariableOp?
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
conv1d_2/BiasAdd?
IdentityIdentityconv1d_2/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/conv1d/ExpandDims_1/ReadVariableOp3^embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp0^embed_seq_2/batch_normalization/AssignMovingAvg?^embed_seq_2/batch_normalization/AssignMovingAvg/ReadVariableOp2^embed_seq_2/batch_normalization/AssignMovingAvg_1A^embed_seq_2/batch_normalization/AssignMovingAvg_1/ReadVariableOp9^embed_seq_2/batch_normalization/batchnorm/ReadVariableOp=^embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp2^embed_seq_2/batch_normalization_1/AssignMovingAvgA^embed_seq_2/batch_normalization_1/AssignMovingAvg/ReadVariableOp4^embed_seq_2/batch_normalization_1/AssignMovingAvg_1C^embed_seq_2/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp;^embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp?^embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp2^embed_seq_2/batch_normalization_2/AssignMovingAvgA^embed_seq_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp4^embed_seq_2/batch_normalization_2/AssignMovingAvg_1C^embed_seq_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp;^embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp?^embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*^embed_seq_2/conv1d/BiasAdd/ReadVariableOp6^embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp,^embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp8^embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp+conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2h
2embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp2embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp2b
/embed_seq_2/batch_normalization/AssignMovingAvg/embed_seq_2/batch_normalization/AssignMovingAvg2?
>embed_seq_2/batch_normalization/AssignMovingAvg/ReadVariableOp>embed_seq_2/batch_normalization/AssignMovingAvg/ReadVariableOp2f
1embed_seq_2/batch_normalization/AssignMovingAvg_11embed_seq_2/batch_normalization/AssignMovingAvg_12?
@embed_seq_2/batch_normalization/AssignMovingAvg_1/ReadVariableOp@embed_seq_2/batch_normalization/AssignMovingAvg_1/ReadVariableOp2t
8embed_seq_2/batch_normalization/batchnorm/ReadVariableOp8embed_seq_2/batch_normalization/batchnorm/ReadVariableOp2|
<embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp<embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp2f
1embed_seq_2/batch_normalization_1/AssignMovingAvg1embed_seq_2/batch_normalization_1/AssignMovingAvg2?
@embed_seq_2/batch_normalization_1/AssignMovingAvg/ReadVariableOp@embed_seq_2/batch_normalization_1/AssignMovingAvg/ReadVariableOp2j
3embed_seq_2/batch_normalization_1/AssignMovingAvg_13embed_seq_2/batch_normalization_1/AssignMovingAvg_12?
Bembed_seq_2/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpBembed_seq_2/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2x
:embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp:embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp2?
>embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp>embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp2f
1embed_seq_2/batch_normalization_2/AssignMovingAvg1embed_seq_2/batch_normalization_2/AssignMovingAvg2?
@embed_seq_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp@embed_seq_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp2j
3embed_seq_2/batch_normalization_2/AssignMovingAvg_13embed_seq_2/batch_normalization_2/AssignMovingAvg_12?
Bembed_seq_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpBembed_seq_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2x
:embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp:embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp2?
>embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp>embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp2V
)embed_seq_2/conv1d/BiasAdd/ReadVariableOp)embed_seq_2/conv1d/BiasAdd/ReadVariableOp2n
5embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp5embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp2Z
+embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp+embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp2r
7embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp7embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_89503

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_91213

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling1d_2_layer_call_fn_91461

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_894632
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90265

inputs(
embed_seq_1_90223:? 
embed_seq_2_90226:	? 
embed_seq_2_90228:	? 
embed_seq_2_90230:	? 
embed_seq_2_90232:	?)
embed_seq_2_90234:?? 
embed_seq_2_90236:	? 
embed_seq_2_90238:	? 
embed_seq_2_90240:	? 
embed_seq_2_90242:	? 
embed_seq_2_90244:	?(
embed_seq_2_90246:?@
embed_seq_2_90248:@
embed_seq_2_90250:@
embed_seq_2_90252:@
embed_seq_2_90254:@
embed_seq_2_90256:@$
conv1d_2_90259:@ 
conv1d_2_90261: 
identity?? conv1d_2/StatefulPartitionedCall?#embed_seq_1/StatefulPartitionedCall?#embed_seq_2/StatefulPartitionedCall?
#embed_seq_1/StatefulPartitionedCallStatefulPartitionedCallinputsembed_seq_1_90223*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_890452%
#embed_seq_1/StatefulPartitionedCall?
#embed_seq_2/StatefulPartitionedCallStatefulPartitionedCall,embed_seq_1/StatefulPartitionedCall:output:0embed_seq_2_90226embed_seq_2_90228embed_seq_2_90230embed_seq_2_90232embed_seq_2_90234embed_seq_2_90236embed_seq_2_90238embed_seq_2_90240embed_seq_2_90242embed_seq_2_90244embed_seq_2_90246embed_seq_2_90248embed_seq_2_90250embed_seq_2_90252embed_seq_2_90254embed_seq_2_90256*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_898992%
#embed_seq_2/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall,embed_seq_2/StatefulPartitionedCall:output:0conv1d_2_90259conv1d_2_90261*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_901172"
 conv1d_2/StatefulPartitionedCall?
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp!^conv1d_2/StatefulPartitionedCall$^embed_seq_1/StatefulPartitionedCall$^embed_seq_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2J
#embed_seq_1/StatefulPartitionedCall#embed_seq_1/StatefulPartitionedCall2J
#embed_seq_2/StatefulPartitionedCall#embed_seq_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
+__inference_embed_seq_1_layer_call_fn_90838

inputs
unknown:?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_890452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
I
-__inference_max_pooling1d_layer_call_fn_91205

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_896552
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????????????:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
A__inference_conv1d_layer_call_and_return_conditional_losses_89682

inputsC
+conv1d_expanddims_1_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2
Relu{
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_91352

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2	
Squeezer
IdentityIdentitySqueeze:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????????????:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?.
?
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_89899

inputs(
batch_normalization_89859:	?(
batch_normalization_89861:	?(
batch_normalization_89863:	?(
batch_normalization_89865:	?$
conv1d_89868:??
conv1d_89870:	?*
batch_normalization_1_89874:	?*
batch_normalization_1_89876:	?*
batch_normalization_1_89878:	?*
batch_normalization_1_89880:	?%
conv1d_1_89883:?@
conv1d_1_89885:@)
batch_normalization_2_89889:@)
batch_normalization_2_89891:@)
batch_normalization_2_89893:@)
batch_normalization_2_89895:@
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_896552
max_pooling1d/PartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0batch_normalization_89859batch_normalization_89861batch_normalization_89863batch_normalization_89865*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_891832-
+batch_normalization/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv1d_89868conv1d_89870*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_896822 
conv1d/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_896952!
max_pooling1d_1/PartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0batch_normalization_1_89874batch_normalization_1_89876batch_normalization_1_89878batch_normalization_1_89880*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_893732/
-batch_normalization_1/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv1d_1_89883conv1d_1_89885*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_897212"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_897342!
max_pooling1d_2/PartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0batch_normalization_2_89889batch_normalization_2_89891batch_normalization_2_89893batch_normalization_2_89895*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_895632/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:???????????????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_1_layer_call_fn_91365

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_893132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling1d_2_layer_call_fn_91466

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_897342
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????@:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
?
A__inference_conv1d_layer_call_and_return_conditional_losses_91326

inputsC
+conv1d_expanddims_1_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2	
BiasAddf
ReluReluBiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2
Relu{
IdentityIdentityRelu:activations:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_91456

inputsB
+conv1d_expanddims_1_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2	
BiasAddx
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_89064
	el1_input 
	el1_89060:?
identity??el1/StatefulPartitionedCall?
el1/StatefulPartitionedCallStatefulPartitionedCall	el1_input	el1_89060*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_el1_layer_call_and_return_conditional_losses_890112
el1/StatefulPartitionedCall?
IdentityIdentity$el1/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityl
NoOpNoOp^el1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 2:
el1/StatefulPartitionedCallel1/StatefulPartitionedCall:_ [
4
_output_shapes"
 :??????????????????
#
_user_specified_name	el1_input
?
?
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_89071
	el1_input 
	el1_89067:?
identity??el1/StatefulPartitionedCall?
el1/StatefulPartitionedCallStatefulPartitionedCall	el1_input	el1_89067*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_el1_layer_call_and_return_conditional_losses_890112
el1/StatefulPartitionedCall?
IdentityIdentity$el1/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityl
NoOpNoOp^el1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 2:
el1/StatefulPartitionedCallel1/StatefulPartitionedCall:_ [
4
_output_shapes"
 :??????????????????
#
_user_specified_name	el1_input
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_91267

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
|
#__inference_el1_layer_call_fn_91183

inputs
unknown:?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_el1_layer_call_and_return_conditional_losses_890112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling1d_1_layer_call_fn_91336

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_896952
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????????????:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
&__inference_conv1d_layer_call_fn_91310

inputs
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_896822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
>__inference_el1_layer_call_and_return_conditional_losses_91195

inputsB
+conv1d_expanddims_1_readvariableop_resource:?
identity??"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2
conv1d/Squeeze?
IdentityIdentityconv1d/Squeeze:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identitys
NoOpNoOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?.
?
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_89746

inputs(
batch_normalization_89657:	?(
batch_normalization_89659:	?(
batch_normalization_89661:	?(
batch_normalization_89663:	?$
conv1d_89683:??
conv1d_89685:	?*
batch_normalization_1_89697:	?*
batch_normalization_1_89699:	?*
batch_normalization_1_89701:	?*
batch_normalization_1_89703:	?%
conv1d_1_89722:?@
conv1d_1_89724:@)
batch_normalization_2_89736:@)
batch_normalization_2_89738:@)
batch_normalization_2_89740:@)
batch_normalization_2_89742:@
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_896552
max_pooling1d/PartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0batch_normalization_89657batch_normalization_89659batch_normalization_89661batch_normalization_89663*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_891232-
+batch_normalization/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv1d_89683conv1d_89685*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_896822 
conv1d/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_896952!
max_pooling1d_1/PartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0batch_normalization_1_89697batch_normalization_1_89699batch_normalization_1_89701batch_normalization_1_89703*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_893132/
-batch_normalization_1/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv1d_1_89722conv1d_1_89724*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_897212"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_897342!
max_pooling1d_2/PartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0batch_normalization_2_89736batch_normalization_2_89738batch_normalization_2_89740batch_normalization_2_89742*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_895032/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:???????????????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_91482

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*8
_output_shapes&
$:"??????????????????@*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2	
Squeezeq
IdentityIdentitySqueeze:output:0*
T0*4
_output_shapes"
 :??????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????????????@:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
?
>__inference_el1_layer_call_and_return_conditional_losses_89011

inputsB
+conv1d_expanddims_1_readvariableop_resource:?
identity??"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2
conv1d/Squeeze?
IdentityIdentityconv1d/Squeeze:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identitys
NoOpNoOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling1d_1_layer_call_fn_91331

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_892732
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_91398

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_91528

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2
batchnorm/add_1{
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
?
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_89016

inputs 
	el1_89012:?
identity??el1/StatefulPartitionedCall?
el1/StatefulPartitionedCallStatefulPartitionedCallinputs	el1_89012*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_el1_layer_call_and_return_conditional_losses_890112
el1/StatefulPartitionedCall?
IdentityIdentity$el1/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityl
NoOpNoOp^el1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 2:
el1/StatefulPartitionedCallel1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_91023

inputsD
5batch_normalization_batchnorm_readvariableop_resource:	?H
9batch_normalization_batchnorm_mul_readvariableop_resource:	?F
7batch_normalization_batchnorm_readvariableop_1_resource:	?F
7batch_normalization_batchnorm_readvariableop_2_resource:	?J
2conv1d_conv1d_expanddims_1_readvariableop_resource:??5
&conv1d_biasadd_readvariableop_resource:	?F
7batch_normalization_1_batchnorm_readvariableop_resource:	?J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	?H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	?H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	?K
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:?@6
(conv1d_1_biasadd_readvariableop_resource:@E
7batch_normalization_2_batchnorm_readvariableop_resource:@I
;batch_normalization_2_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_2_batchnorm_readvariableop_1_resource:@G
9batch_normalization_2_batchnorm_readvariableop_2_resource:@
identity??,batch_normalization/batchnorm/ReadVariableOp?.batch_normalization/batchnorm/ReadVariableOp_1?.batch_normalization/batchnorm/ReadVariableOp_2?0batch_normalization/batchnorm/mul/ReadVariableOp?.batch_normalization_1/batchnorm/ReadVariableOp?0batch_normalization_1/batchnorm/ReadVariableOp_1?0batch_normalization_1/batchnorm/ReadVariableOp_2?2batch_normalization_1/batchnorm/mul/ReadVariableOp?.batch_normalization_2/batchnorm/ReadVariableOp?0batch_normalization_2/batchnorm/ReadVariableOp_1?0batch_normalization_2/batchnorm/ReadVariableOp_2?2batch_normalization_2/batchnorm/mul/ReadVariableOp?conv1d/BiasAdd/ReadVariableOp?)conv1d/conv1d/ExpandDims_1/ReadVariableOp?conv1d_1/BiasAdd/ReadVariableOp?+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp~
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
max_pooling1d/ExpandDims/dim?
max_pooling1d/ExpandDims
ExpandDimsinputs%max_pooling1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
max_pooling1d/ExpandDims?
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d/MaxPool?
max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
max_pooling1d/Squeeze?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/Rsqrt?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Mulmax_pooling1d/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2%
#batch_normalization/batchnorm/mul_1?
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1?
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/mul_2?
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2?
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2%
#batch_normalization/batchnorm/add_1?
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/conv1d/ExpandDims/dim?
conv1d/conv1d/ExpandDims
ExpandDims'batch_normalization/batchnorm/add_1:z:0%conv1d/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d/conv1d/ExpandDims?
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOp?
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim?
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??2
conv1d/conv1d/ExpandDims_1?
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2
conv1d/conv1d?
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2
conv1d/conv1d/Squeeze?
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv1d/BiasAdd/ReadVariableOp?
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2
conv1d/BiasAdd{
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2
conv1d/Relu?
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_1/ExpandDims/dim?
max_pooling1d_1/ExpandDims
ExpandDimsconv1d/Relu:activations:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
max_pooling1d_1/ExpandDims?
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2
max_pooling1d_1/MaxPool?
max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2
max_pooling1d_1/Squeeze?
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp?
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_1/batchnorm/add/y?
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/add?
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/Rsqrt?
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp?
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/mul?
%batch_normalization_1/batchnorm/mul_1Mul max_pooling1d_1/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2'
%batch_normalization_1/batchnorm/mul_1?
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1?
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2'
%batch_normalization_1/batchnorm/mul_2?
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2?
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization_1/batchnorm/sub?
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2'
%batch_normalization_1/batchnorm/add_1?
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2 
conv1d_1/conv1d/ExpandDims/dim?
conv1d_1/conv1d/ExpandDims
ExpandDims)batch_normalization_1/batchnorm/add_1:z:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2
conv1d_1/conv1d/ExpandDims?
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?@*
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim?
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@2
conv1d_1/conv1d/ExpandDims_1?
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2
conv1d_1/conv1d?
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims

?????????2
conv1d_1/conv1d/Squeeze?
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_1/BiasAdd/ReadVariableOp?
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2
conv1d_1/BiasAdd?
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2 
max_pooling1d_2/ExpandDims/dim?
max_pooling1d_2/ExpandDims
ExpandDimsconv1d_1/BiasAdd:output:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
max_pooling1d_2/ExpandDims?
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*8
_output_shapes&
$:"??????????????????@*
ksize
*
paddingVALID*
strides
2
max_pooling1d_2/MaxPool?
max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2
max_pooling1d_2/Squeeze?
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp?
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2'
%batch_normalization_2/batchnorm/add/y?
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/add?
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrt?
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp?
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul?
%batch_normalization_2/batchnorm/mul_1Mul max_pooling1d_2/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2'
%batch_normalization_2/batchnorm/mul_1?
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1?
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2?
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2?
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/sub?
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2'
%batch_normalization_2/batchnorm/add_1?
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/conv1d/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:???????????????????: : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/conv1d/ExpandDims_1/ReadVariableOp)conv1d/conv1d/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp+conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_89045

inputs 
	el1_89041:?
identity??el1/StatefulPartitionedCall?
el1/StatefulPartitionedCallStatefulPartitionedCallinputs	el1_89041*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_el1_layer_call_and_return_conditional_losses_890112
el1/StatefulPartitionedCall?
IdentityIdentity$el1/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityl
NoOpNoOp^el1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 2:
el1/StatefulPartitionedCallel1/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv1d_2_layer_call_fn_91161

inputs
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_901172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90439
ipt(
embed_seq_1_90397:? 
embed_seq_2_90400:	? 
embed_seq_2_90402:	? 
embed_seq_2_90404:	? 
embed_seq_2_90406:	?)
embed_seq_2_90408:?? 
embed_seq_2_90410:	? 
embed_seq_2_90412:	? 
embed_seq_2_90414:	? 
embed_seq_2_90416:	? 
embed_seq_2_90418:	?(
embed_seq_2_90420:?@
embed_seq_2_90422:@
embed_seq_2_90424:@
embed_seq_2_90426:@
embed_seq_2_90428:@
embed_seq_2_90430:@$
conv1d_2_90433:@ 
conv1d_2_90435: 
identity?? conv1d_2/StatefulPartitionedCall?#embed_seq_1/StatefulPartitionedCall?#embed_seq_2/StatefulPartitionedCall?
#embed_seq_1/StatefulPartitionedCallStatefulPartitionedCalliptembed_seq_1_90397*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_890452%
#embed_seq_1/StatefulPartitionedCall?
#embed_seq_2/StatefulPartitionedCallStatefulPartitionedCall,embed_seq_1/StatefulPartitionedCall:output:0embed_seq_2_90400embed_seq_2_90402embed_seq_2_90404embed_seq_2_90406embed_seq_2_90408embed_seq_2_90410embed_seq_2_90412embed_seq_2_90414embed_seq_2_90416embed_seq_2_90418embed_seq_2_90420embed_seq_2_90422embed_seq_2_90424embed_seq_2_90426embed_seq_2_90428embed_seq_2_90430*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_898992%
#embed_seq_2/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall,embed_seq_2/StatefulPartitionedCall:output:0conv1d_2_90433conv1d_2_90435*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_901172"
 conv1d_2/StatefulPartitionedCall?
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp!^conv1d_2/StatefulPartitionedCall$^embed_seq_1/StatefulPartitionedCall$^embed_seq_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2J
#embed_seq_1/StatefulPartitionedCall#embed_seq_1/StatefulPartitionedCall2J
#embed_seq_2/StatefulPartitionedCall#embed_seq_2/StatefulPartitionedCall:Y U
4
_output_shapes"
 :??????????????????

_user_specified_nameipt
?1
?	
__inference__traced_save_91642
file_prefix.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop)
%savev2_el1_kernel_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop%savev2_el1_kernel_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@ : :?:?:??:?:?:?:?@:@:@:@:?:?:?:?:?:@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
:@ : 

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:*&
$
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:)	%
#
_output_shapes
:?@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:)%
#
_output_shapes
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?: 

_output_shapes
:@: 

_output_shapes
:@:

_output_shapes
: 
?
?
+__inference_embed_seq_1_layer_call_fn_90831

inputs
unknown:?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_890162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
)__inference_embed_seq_layer_call_fn_90527

inputs
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?!
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?!

unknown_10:?@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16:@ 

unknown_17: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_embed_seq_layer_call_and_return_conditional_losses_901242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_89463

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90394
ipt(
embed_seq_1_90352:? 
embed_seq_2_90355:	? 
embed_seq_2_90357:	? 
embed_seq_2_90359:	? 
embed_seq_2_90361:	?)
embed_seq_2_90363:?? 
embed_seq_2_90365:	? 
embed_seq_2_90367:	? 
embed_seq_2_90369:	? 
embed_seq_2_90371:	? 
embed_seq_2_90373:	?(
embed_seq_2_90375:?@
embed_seq_2_90377:@
embed_seq_2_90379:@
embed_seq_2_90381:@
embed_seq_2_90383:@
embed_seq_2_90385:@$
conv1d_2_90388:@ 
conv1d_2_90390: 
identity?? conv1d_2/StatefulPartitionedCall?#embed_seq_1/StatefulPartitionedCall?#embed_seq_2/StatefulPartitionedCall?
#embed_seq_1/StatefulPartitionedCallStatefulPartitionedCalliptembed_seq_1_90352*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_890162%
#embed_seq_1/StatefulPartitionedCall?
#embed_seq_2/StatefulPartitionedCallStatefulPartitionedCall,embed_seq_1/StatefulPartitionedCall:output:0embed_seq_2_90355embed_seq_2_90357embed_seq_2_90359embed_seq_2_90361embed_seq_2_90363embed_seq_2_90365embed_seq_2_90367embed_seq_2_90369embed_seq_2_90371embed_seq_2_90373embed_seq_2_90375embed_seq_2_90377embed_seq_2_90379embed_seq_2_90381embed_seq_2_90383embed_seq_2_90385*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_897462%
#embed_seq_2/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall,embed_seq_2/StatefulPartitionedCall:output:0conv1d_2_90388conv1d_2_90390*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_901172"
 conv1d_2/StatefulPartitionedCall?
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp!^conv1d_2/StatefulPartitionedCall$^embed_seq_1/StatefulPartitionedCall$^embed_seq_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2J
#embed_seq_1/StatefulPartitionedCall#embed_seq_1/StatefulPartitionedCall2J
#embed_seq_2/StatefulPartitionedCall#embed_seq_2/StatefulPartitionedCall:Y U
4
_output_shapes"
 :??????????????????

_user_specified_nameipt
?.
?
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_90059
max_pooling1d_input(
batch_normalization_90019:	?(
batch_normalization_90021:	?(
batch_normalization_90023:	?(
batch_normalization_90025:	?$
conv1d_90028:??
conv1d_90030:	?*
batch_normalization_1_90034:	?*
batch_normalization_1_90036:	?*
batch_normalization_1_90038:	?*
batch_normalization_1_90040:	?%
conv1d_1_90043:?@
conv1d_1_90045:@)
batch_normalization_2_90049:@)
batch_normalization_2_90051:@)
batch_normalization_2_90053:@)
batch_normalization_2_90055:@
identity??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?-batch_normalization_2/StatefulPartitionedCall?conv1d/StatefulPartitionedCall? conv1d_1/StatefulPartitionedCall?
max_pooling1d/PartitionedCallPartitionedCallmax_pooling1d_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_896552
max_pooling1d/PartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling1d/PartitionedCall:output:0batch_normalization_90019batch_normalization_90021batch_normalization_90023batch_normalization_90025*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_891832-
+batch_normalization/StatefulPartitionedCall?
conv1d/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv1d_90028conv1d_90030*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_896822 
conv1d/StatefulPartitionedCall?
max_pooling1d_1/PartitionedCallPartitionedCall'conv1d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_896952!
max_pooling1d_1/PartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_1/PartitionedCall:output:0batch_normalization_1_90034batch_normalization_1_90036batch_normalization_1_90038batch_normalization_1_90040*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_893732/
-batch_normalization_1/StatefulPartitionedCall?
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv1d_1_90043conv1d_1_90045*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_897212"
 conv1d_1/StatefulPartitionedCall?
max_pooling1d_2/PartitionedCallPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_897342!
max_pooling1d_2/PartitionedCall?
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling1d_2/PartitionedCall:output:0batch_normalization_2_90049batch_normalization_2_90051batch_normalization_2_90053batch_normalization_2_90055*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_895632/
-batch_normalization_2/StatefulPartitionedCall?
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:???????????????????: : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall:j f
5
_output_shapes#
!:???????????????????
-
_user_specified_namemax_pooling1d_input
?
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_89273

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_91176

inputsA
+conv1d_expanddims_1_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? *
paddingSAME*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*4
_output_shapes"
 :?????????????????? *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2	
BiasAddx
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
?
?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90124

inputs(
embed_seq_1_90066:? 
embed_seq_2_90069:	? 
embed_seq_2_90071:	? 
embed_seq_2_90073:	? 
embed_seq_2_90075:	?)
embed_seq_2_90077:?? 
embed_seq_2_90079:	? 
embed_seq_2_90081:	? 
embed_seq_2_90083:	? 
embed_seq_2_90085:	? 
embed_seq_2_90087:	?(
embed_seq_2_90089:?@
embed_seq_2_90091:@
embed_seq_2_90093:@
embed_seq_2_90095:@
embed_seq_2_90097:@
embed_seq_2_90099:@$
conv1d_2_90118:@ 
conv1d_2_90120: 
identity?? conv1d_2/StatefulPartitionedCall?#embed_seq_1/StatefulPartitionedCall?#embed_seq_2/StatefulPartitionedCall?
#embed_seq_1/StatefulPartitionedCallStatefulPartitionedCallinputsembed_seq_1_90066*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_890162%
#embed_seq_1/StatefulPartitionedCall?
#embed_seq_2/StatefulPartitionedCallStatefulPartitionedCall,embed_seq_1/StatefulPartitionedCall:output:0embed_seq_2_90069embed_seq_2_90071embed_seq_2_90073embed_seq_2_90075embed_seq_2_90077embed_seq_2_90079embed_seq_2_90081embed_seq_2_90083embed_seq_2_90085embed_seq_2_90087embed_seq_2_90089embed_seq_2_90091embed_seq_2_90093embed_seq_2_90095embed_seq_2_90097embed_seq_2_90099*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_897462%
#embed_seq_2/StatefulPartitionedCall?
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall,embed_seq_2/StatefulPartitionedCall:output:0conv1d_2_90118conv1d_2_90120*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_901172"
 conv1d_2/StatefulPartitionedCall?
IdentityIdentity)conv1d_2/StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp!^conv1d_2/StatefulPartitionedCall$^embed_seq_1/StatefulPartitionedCall$^embed_seq_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2J
#embed_seq_1/StatefulPartitionedCall#embed_seq_1/StatefulPartitionedCall2J
#embed_seq_2/StatefulPartitionedCall#embed_seq_2/StatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_88992
ipt\
Eembed_seq_embed_seq_1_el1_conv1d_expanddims_1_readvariableop_resource:?Z
Kembed_seq_embed_seq_2_batch_normalization_batchnorm_readvariableop_resource:	?^
Oembed_seq_embed_seq_2_batch_normalization_batchnorm_mul_readvariableop_resource:	?\
Membed_seq_embed_seq_2_batch_normalization_batchnorm_readvariableop_1_resource:	?\
Membed_seq_embed_seq_2_batch_normalization_batchnorm_readvariableop_2_resource:	?`
Hembed_seq_embed_seq_2_conv1d_conv1d_expanddims_1_readvariableop_resource:??K
<embed_seq_embed_seq_2_conv1d_biasadd_readvariableop_resource:	?\
Membed_seq_embed_seq_2_batch_normalization_1_batchnorm_readvariableop_resource:	?`
Qembed_seq_embed_seq_2_batch_normalization_1_batchnorm_mul_readvariableop_resource:	?^
Oembed_seq_embed_seq_2_batch_normalization_1_batchnorm_readvariableop_1_resource:	?^
Oembed_seq_embed_seq_2_batch_normalization_1_batchnorm_readvariableop_2_resource:	?a
Jembed_seq_embed_seq_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource:?@L
>embed_seq_embed_seq_2_conv1d_1_biasadd_readvariableop_resource:@[
Membed_seq_embed_seq_2_batch_normalization_2_batchnorm_readvariableop_resource:@_
Qembed_seq_embed_seq_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:@]
Oembed_seq_embed_seq_2_batch_normalization_2_batchnorm_readvariableop_1_resource:@]
Oembed_seq_embed_seq_2_batch_normalization_2_batchnorm_readvariableop_2_resource:@T
>embed_seq_conv1d_2_conv1d_expanddims_1_readvariableop_resource:@ @
2embed_seq_conv1d_2_biasadd_readvariableop_resource: 
identity??)embed_seq/conv1d_2/BiasAdd/ReadVariableOp?5embed_seq/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?<embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp?Bembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp?Dembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1?Dembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2?Fembed_seq/embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp?Dembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp?Fembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1?Fembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2?Hembed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp?Dembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp?Fembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1?Fembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2?Hembed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp?3embed_seq/embed_seq_2/conv1d/BiasAdd/ReadVariableOp??embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp?5embed_seq/embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp?Aembed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
/embed_seq/embed_seq_1/el1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????21
/embed_seq/embed_seq_1/el1/conv1d/ExpandDims/dim?
+embed_seq/embed_seq_1/el1/conv1d/ExpandDims
ExpandDimsipt8embed_seq/embed_seq_1/el1/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2-
+embed_seq/embed_seq_1/el1/conv1d/ExpandDims?
<embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpEembed_seq_embed_seq_1_el1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02>
<embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp?
1embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1/dim?
-embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1
ExpandDimsDembed_seq/embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp:value:0:embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2/
-embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1?
 embed_seq/embed_seq_1/el1/conv1dConv2D4embed_seq/embed_seq_1/el1/conv1d/ExpandDims:output:06embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2"
 embed_seq/embed_seq_1/el1/conv1d?
(embed_seq/embed_seq_1/el1/conv1d/SqueezeSqueeze)embed_seq/embed_seq_1/el1/conv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2*
(embed_seq/embed_seq_1/el1/conv1d/Squeeze?
2embed_seq/embed_seq_2/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2embed_seq/embed_seq_2/max_pooling1d/ExpandDims/dim?
.embed_seq/embed_seq_2/max_pooling1d/ExpandDims
ExpandDims1embed_seq/embed_seq_1/el1/conv1d/Squeeze:output:0;embed_seq/embed_seq_2/max_pooling1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????20
.embed_seq/embed_seq_2/max_pooling1d/ExpandDims?
+embed_seq/embed_seq_2/max_pooling1d/MaxPoolMaxPool7embed_seq/embed_seq_2/max_pooling1d/ExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2-
+embed_seq/embed_seq_2/max_pooling1d/MaxPool?
+embed_seq/embed_seq_2/max_pooling1d/SqueezeSqueeze4embed_seq/embed_seq_2/max_pooling1d/MaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2-
+embed_seq/embed_seq_2/max_pooling1d/Squeeze?
Bembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOpReadVariableOpKembed_seq_embed_seq_2_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02D
Bembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp?
9embed_seq/embed_seq_2/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2;
9embed_seq/embed_seq_2/batch_normalization/batchnorm/add/y?
7embed_seq/embed_seq_2/batch_normalization/batchnorm/addAddV2Jembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp:value:0Bembed_seq/embed_seq_2/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?29
7embed_seq/embed_seq_2/batch_normalization/batchnorm/add?
9embed_seq/embed_seq_2/batch_normalization/batchnorm/RsqrtRsqrt;embed_seq/embed_seq_2/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?2;
9embed_seq/embed_seq_2/batch_normalization/batchnorm/Rsqrt?
Fembed_seq/embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpOembed_seq_embed_seq_2_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02H
Fembed_seq/embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp?
7embed_seq/embed_seq_2/batch_normalization/batchnorm/mulMul=embed_seq/embed_seq_2/batch_normalization/batchnorm/Rsqrt:y:0Nembed_seq/embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?29
7embed_seq/embed_seq_2/batch_normalization/batchnorm/mul?
9embed_seq/embed_seq_2/batch_normalization/batchnorm/mul_1Mul4embed_seq/embed_seq_2/max_pooling1d/Squeeze:output:0;embed_seq/embed_seq_2/batch_normalization/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2;
9embed_seq/embed_seq_2/batch_normalization/batchnorm/mul_1?
Dembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpMembed_seq_embed_seq_2_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02F
Dembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1?
9embed_seq/embed_seq_2/batch_normalization/batchnorm/mul_2MulLembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1:value:0;embed_seq/embed_seq_2/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2;
9embed_seq/embed_seq_2/batch_normalization/batchnorm/mul_2?
Dembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpMembed_seq_embed_seq_2_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02F
Dembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2?
7embed_seq/embed_seq_2/batch_normalization/batchnorm/subSubLembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2:value:0=embed_seq/embed_seq_2/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?29
7embed_seq/embed_seq_2/batch_normalization/batchnorm/sub?
9embed_seq/embed_seq_2/batch_normalization/batchnorm/add_1AddV2=embed_seq/embed_seq_2/batch_normalization/batchnorm/mul_1:z:0;embed_seq/embed_seq_2/batch_normalization/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2;
9embed_seq/embed_seq_2/batch_normalization/batchnorm/add_1?
2embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims/dim?
.embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims
ExpandDims=embed_seq/embed_seq_2/batch_normalization/batchnorm/add_1:z:0;embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????20
.embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims?
?embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHembed_seq_embed_seq_2_conv1d_conv1d_expanddims_1_readvariableop_resource*$
_output_shapes
:??*
dtype02A
?embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp?
4embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1/dim?
0embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1
ExpandDimsGembed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0=embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*(
_output_shapes
:??22
0embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1?
#embed_seq/embed_seq_2/conv1d/conv1dConv2D7embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims:output:09embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2%
#embed_seq/embed_seq_2/conv1d/conv1d?
+embed_seq/embed_seq_2/conv1d/conv1d/SqueezeSqueeze,embed_seq/embed_seq_2/conv1d/conv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2-
+embed_seq/embed_seq_2/conv1d/conv1d/Squeeze?
3embed_seq/embed_seq_2/conv1d/BiasAdd/ReadVariableOpReadVariableOp<embed_seq_embed_seq_2_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype025
3embed_seq/embed_seq_2/conv1d/BiasAdd/ReadVariableOp?
$embed_seq/embed_seq_2/conv1d/BiasAddBiasAdd4embed_seq/embed_seq_2/conv1d/conv1d/Squeeze:output:0;embed_seq/embed_seq_2/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:???????????????????2&
$embed_seq/embed_seq_2/conv1d/BiasAdd?
!embed_seq/embed_seq_2/conv1d/ReluRelu-embed_seq/embed_seq_2/conv1d/BiasAdd:output:0*
T0*5
_output_shapes#
!:???????????????????2#
!embed_seq/embed_seq_2/conv1d/Relu?
4embed_seq/embed_seq_2/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :26
4embed_seq/embed_seq_2/max_pooling1d_1/ExpandDims/dim?
0embed_seq/embed_seq_2/max_pooling1d_1/ExpandDims
ExpandDims/embed_seq/embed_seq_2/conv1d/Relu:activations:0=embed_seq/embed_seq_2/max_pooling1d_1/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????22
0embed_seq/embed_seq_2/max_pooling1d_1/ExpandDims?
-embed_seq/embed_seq_2/max_pooling1d_1/MaxPoolMaxPool9embed_seq/embed_seq_2/max_pooling1d_1/ExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2/
-embed_seq/embed_seq_2/max_pooling1d_1/MaxPool?
-embed_seq/embed_seq_2/max_pooling1d_1/SqueezeSqueeze6embed_seq/embed_seq_2/max_pooling1d_1/MaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2/
-embed_seq/embed_seq_2/max_pooling1d_1/Squeeze?
Dembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpMembed_seq_embed_seq_2_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02F
Dembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp?
;embed_seq/embed_seq_2/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2=
;embed_seq/embed_seq_2/batch_normalization_1/batchnorm/add/y?
9embed_seq/embed_seq_2/batch_normalization_1/batchnorm/addAddV2Lembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp:value:0Dembed_seq/embed_seq_2/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2;
9embed_seq/embed_seq_2/batch_normalization_1/batchnorm/add?
;embed_seq/embed_seq_2/batch_normalization_1/batchnorm/RsqrtRsqrt=embed_seq/embed_seq_2/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:?2=
;embed_seq/embed_seq_2/batch_normalization_1/batchnorm/Rsqrt?
Hembed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpQembed_seq_embed_seq_2_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02J
Hembed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp?
9embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mulMul?embed_seq/embed_seq_2/batch_normalization_1/batchnorm/Rsqrt:y:0Pembed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2;
9embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul?
;embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul_1Mul6embed_seq/embed_seq_2/max_pooling1d_1/Squeeze:output:0=embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2=
;embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul_1?
Fembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpOembed_seq_embed_seq_2_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02H
Fembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1?
;embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul_2MulNembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0=embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2=
;embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul_2?
Fembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpOembed_seq_embed_seq_2_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02H
Fembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2?
9embed_seq/embed_seq_2/batch_normalization_1/batchnorm/subSubNembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0?embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2;
9embed_seq/embed_seq_2/batch_normalization_1/batchnorm/sub?
;embed_seq/embed_seq_2/batch_normalization_1/batchnorm/add_1AddV2?embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul_1:z:0=embed_seq/embed_seq_2/batch_normalization_1/batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2=
;embed_seq/embed_seq_2/batch_normalization_1/batchnorm/add_1?
4embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????26
4embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims/dim?
0embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims
ExpandDims?embed_seq/embed_seq_2/batch_normalization_1/batchnorm/add_1:z:0=embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????22
0embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims?
Aembed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpJembed_seq_embed_seq_2_conv1d_1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?@*
dtype02C
Aembed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?
6embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 28
6embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1/dim?
2embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1
ExpandDimsIembed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0?embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?@24
2embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1?
%embed_seq/embed_seq_2/conv1d_1/conv1dConv2D9embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims:output:0;embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"??????????????????@*
paddingSAME*
strides
2'
%embed_seq/embed_seq_2/conv1d_1/conv1d?
-embed_seq/embed_seq_2/conv1d_1/conv1d/SqueezeSqueeze.embed_seq/embed_seq_2/conv1d_1/conv1d:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims

?????????2/
-embed_seq/embed_seq_2/conv1d_1/conv1d/Squeeze?
5embed_seq/embed_seq_2/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp>embed_seq_embed_seq_2_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5embed_seq/embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp?
&embed_seq/embed_seq_2/conv1d_1/BiasAddBiasAdd6embed_seq/embed_seq_2/conv1d_1/conv1d/Squeeze:output:0=embed_seq/embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????@2(
&embed_seq/embed_seq_2/conv1d_1/BiasAdd?
4embed_seq/embed_seq_2/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :26
4embed_seq/embed_seq_2/max_pooling1d_2/ExpandDims/dim?
0embed_seq/embed_seq_2/max_pooling1d_2/ExpandDims
ExpandDims/embed_seq/embed_seq_2/conv1d_1/BiasAdd:output:0=embed_seq/embed_seq_2/max_pooling1d_2/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@22
0embed_seq/embed_seq_2/max_pooling1d_2/ExpandDims?
-embed_seq/embed_seq_2/max_pooling1d_2/MaxPoolMaxPool9embed_seq/embed_seq_2/max_pooling1d_2/ExpandDims:output:0*8
_output_shapes&
$:"??????????????????@*
ksize
*
paddingVALID*
strides
2/
-embed_seq/embed_seq_2/max_pooling1d_2/MaxPool?
-embed_seq/embed_seq_2/max_pooling1d_2/SqueezeSqueeze6embed_seq/embed_seq_2/max_pooling1d_2/MaxPool:output:0*
T0*4
_output_shapes"
 :??????????????????@*
squeeze_dims
2/
-embed_seq/embed_seq_2/max_pooling1d_2/Squeeze?
Dembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpMembed_seq_embed_seq_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
Dembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp?
;embed_seq/embed_seq_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2=
;embed_seq/embed_seq_2/batch_normalization_2/batchnorm/add/y?
9embed_seq/embed_seq_2/batch_normalization_2/batchnorm/addAddV2Lembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0Dembed_seq/embed_seq_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2;
9embed_seq/embed_seq_2/batch_normalization_2/batchnorm/add?
;embed_seq/embed_seq_2/batch_normalization_2/batchnorm/RsqrtRsqrt=embed_seq/embed_seq_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2=
;embed_seq/embed_seq_2/batch_normalization_2/batchnorm/Rsqrt?
Hembed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpQembed_seq_embed_seq_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
Hembed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp?
9embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mulMul?embed_seq/embed_seq_2/batch_normalization_2/batchnorm/Rsqrt:y:0Pembed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2;
9embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul?
;embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul_1Mul6embed_seq/embed_seq_2/max_pooling1d_2/Squeeze:output:0=embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@2=
;embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul_1?
Fembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpOembed_seq_embed_seq_2_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
Fembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1?
;embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul_2MulNembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0=embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2=
;embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul_2?
Fembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpOembed_seq_embed_seq_2_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02H
Fembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2?
9embed_seq/embed_seq_2/batch_normalization_2/batchnorm/subSubNembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0?embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2;
9embed_seq/embed_seq_2/batch_normalization_2/batchnorm/sub?
;embed_seq/embed_seq_2/batch_normalization_2/batchnorm/add_1AddV2?embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul_1:z:0=embed_seq/embed_seq_2/batch_normalization_2/batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@2=
;embed_seq/embed_seq_2/batch_normalization_2/batchnorm/add_1?
(embed_seq/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(embed_seq/conv1d_2/conv1d/ExpandDims/dim?
$embed_seq/conv1d_2/conv1d/ExpandDims
ExpandDims?embed_seq/embed_seq_2/batch_normalization_2/batchnorm/add_1:z:01embed_seq/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????@2&
$embed_seq/conv1d_2/conv1d/ExpandDims?
5embed_seq/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp>embed_seq_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype027
5embed_seq/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?
*embed_seq/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2,
*embed_seq/conv1d_2/conv1d/ExpandDims_1/dim?
&embed_seq/conv1d_2/conv1d/ExpandDims_1
ExpandDims=embed_seq/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:03embed_seq/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2(
&embed_seq/conv1d_2/conv1d/ExpandDims_1?
embed_seq/conv1d_2/conv1dConv2D-embed_seq/conv1d_2/conv1d/ExpandDims:output:0/embed_seq/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*8
_output_shapes&
$:"?????????????????? *
paddingSAME*
strides
2
embed_seq/conv1d_2/conv1d?
!embed_seq/conv1d_2/conv1d/SqueezeSqueeze"embed_seq/conv1d_2/conv1d:output:0*
T0*4
_output_shapes"
 :?????????????????? *
squeeze_dims

?????????2#
!embed_seq/conv1d_2/conv1d/Squeeze?
)embed_seq/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp2embed_seq_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)embed_seq/conv1d_2/BiasAdd/ReadVariableOp?
embed_seq/conv1d_2/BiasAddBiasAdd*embed_seq/conv1d_2/conv1d/Squeeze:output:01embed_seq/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
embed_seq/conv1d_2/BiasAdd?
IdentityIdentity#embed_seq/conv1d_2/BiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?

NoOpNoOp*^embed_seq/conv1d_2/BiasAdd/ReadVariableOp6^embed_seq/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp=^embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOpC^embed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOpE^embed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1E^embed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2G^embed_seq/embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOpE^embed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOpG^embed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1G^embed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2I^embed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOpE^embed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOpG^embed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1G^embed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2I^embed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp4^embed_seq/embed_seq_2/conv1d/BiasAdd/ReadVariableOp@^embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp6^embed_seq/embed_seq_2/conv1d_1/BiasAdd/ReadVariableOpB^embed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 2V
)embed_seq/conv1d_2/BiasAdd/ReadVariableOp)embed_seq/conv1d_2/BiasAdd/ReadVariableOp2n
5embed_seq/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp5embed_seq/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2|
<embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp<embed_seq/embed_seq_1/el1/conv1d/ExpandDims_1/ReadVariableOp2?
Bembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOpBembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp2?
Dembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_1Dembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_12?
Dembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_2Dembed_seq/embed_seq_2/batch_normalization/batchnorm/ReadVariableOp_22?
Fembed_seq/embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOpFembed_seq/embed_seq_2/batch_normalization/batchnorm/mul/ReadVariableOp2?
Dembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOpDembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp2?
Fembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_1Fembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_12?
Fembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_2Fembed_seq/embed_seq_2/batch_normalization_1/batchnorm/ReadVariableOp_22?
Hembed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOpHembed_seq/embed_seq_2/batch_normalization_1/batchnorm/mul/ReadVariableOp2?
Dembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOpDembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp2?
Fembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_1Fembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_12?
Fembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_2Fembed_seq/embed_seq_2/batch_normalization_2/batchnorm/ReadVariableOp_22?
Hembed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOpHembed_seq/embed_seq_2/batch_normalization_2/batchnorm/mul/ReadVariableOp2j
3embed_seq/embed_seq_2/conv1d/BiasAdd/ReadVariableOp3embed_seq/embed_seq_2/conv1d/BiasAdd/ReadVariableOp2?
?embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp?embed_seq/embed_seq_2/conv1d/conv1d/ExpandDims_1/ReadVariableOp2n
5embed_seq/embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp5embed_seq/embed_seq_2/conv1d_1/BiasAdd/ReadVariableOp2?
Aembed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpAembed_seq/embed_seq_2/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:Y U
4
_output_shapes"
 :??????????????????

_user_specified_nameipt
?
?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_89313

inputs0
!batchnorm_readvariableop_resource:	?4
%batchnorm_mul_readvariableop_resource:	?2
#batchnorm_readvariableop_1_resource:	?2
#batchnorm_readvariableop_2_resource:	?
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1|
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identity?
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):???????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
)__inference_embed_seq_layer_call_fn_90165
ipt
unknown:?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?!
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?
	unknown_9:	?!

unknown_10:?@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@ 

unknown_16:@ 

unknown_17: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalliptunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_embed_seq_layer_call_and_return_conditional_losses_901242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:??????????????????: : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
4
_output_shapes"
 :??????????????????

_user_specified_nameipt
?
?
+__inference_embed_seq_2_layer_call_fn_90899

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?!
	unknown_3:??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	? 
	unknown_9:?@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_897462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:???????????????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_89695

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*9
_output_shapes'
%:#???????????????????2

ExpandDims?
MaxPoolMaxPoolExpandDims:output:0*9
_output_shapes'
%:#???????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
SqueezeSqueezeMaxPool:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims
2	
Squeezer
IdentityIdentitySqueeze:output:0*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????????????:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_90850

inputsF
/el1_conv1d_expanddims_1_readvariableop_resource:?
identity??&el1/conv1d/ExpandDims_1/ReadVariableOp?
el1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
el1/conv1d/ExpandDims/dim?
el1/conv1d/ExpandDims
ExpandDimsinputs"el1/conv1d/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"??????????????????2
el1/conv1d/ExpandDims?
&el1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp/el1_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:?*
dtype02(
&el1/conv1d/ExpandDims_1/ReadVariableOp|
el1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
el1/conv1d/ExpandDims_1/dim?
el1/conv1d/ExpandDims_1
ExpandDims.el1/conv1d/ExpandDims_1/ReadVariableOp:value:0$el1/conv1d/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:?2
el1/conv1d/ExpandDims_1?

el1/conv1dConv2Del1/conv1d/ExpandDims:output:0 el1/conv1d/ExpandDims_1:output:0*
T0*9
_output_shapes'
%:#???????????????????*
paddingSAME*
strides
2

el1/conv1d?
el1/conv1d/SqueezeSqueezeel1/conv1d:output:0*
T0*5
_output_shapes#
!:???????????????????*
squeeze_dims

?????????2
el1/conv1d/Squeeze?
IdentityIdentityel1/conv1d/Squeeze:output:0^NoOp*
T0*5
_output_shapes#
!:???????????????????2

Identityw
NoOpNoOp'^el1/conv1d/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????????????: 2P
&el1/conv1d/ExpandDims_1/ReadVariableOp&el1/conv1d/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
ipt9
serving_default_ipt:0??????????????????I
conv1d_2=
StatefulPartitionedCall:0?????????????????? tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

layer_with_weights-0

layer-0
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_sequential
?

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
v
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
10
11"
trackable_list_wrapper
 "
trackable_list_wrapper
?
+0
!1
"2
,3
-4
#5
$6
%7
&8
.9
/10
'11
(12
)13
*14
015
116
17
18"
trackable_list_wrapper
?
2layer_regularization_losses
trainable_variables
3non_trainable_variables
regularization_losses

4layers
	variables
5layer_metrics
6metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

+kernel
7trainable_variables
8regularization_losses
9	variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
?
;layer_regularization_losses
trainable_variables
<non_trainable_variables
regularization_losses

=layers
	variables
>layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Daxis
	!gamma
"beta
,moving_mean
-moving_variance
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

#kernel
$bias
Itrainable_variables
Jregularization_losses
K	variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Qaxis
	%gamma
&beta
.moving_mean
/moving_variance
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

'kernel
(bias
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
^axis
	)gamma
*beta
0moving_mean
1moving_variance
_trainable_variables
`regularization_losses
a	variables
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
f
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
!0
"1
,2
-3
#4
$5
%6
&7
.8
/9
'10
(11
)12
*13
014
115"
trackable_list_wrapper
?
clayer_regularization_losses
trainable_variables
dnon_trainable_variables
regularization_losses

elayers
	variables
flayer_metrics
gmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#@ 2conv1d_2/kernel
: 2conv1d_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
hlayer_regularization_losses
trainable_variables
inon_trainable_variables
regularization_losses

jlayers
	variables
klayer_metrics
lmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&?2batch_normalization/gamma
':%?2batch_normalization/beta
%:#??2conv1d/kernel
:?2conv1d/bias
*:(?2batch_normalization_1/gamma
):'?2batch_normalization_1/beta
&:$?@2conv1d_1/kernel
:@2conv1d_1/bias
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
!:?2
el1/kernel
0:.? (2batch_normalization/moving_mean
4:2? (2#batch_normalization/moving_variance
2:0? (2!batch_normalization_1/moving_mean
6:4? (2%batch_normalization_1/moving_variance
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
Q
+0
,1
-2
.3
/4
05
16"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
?
mlayer_regularization_losses
7trainable_variables
nnon_trainable_variables
8regularization_losses

olayers
9	variables
player_metrics
qmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
rlayer_regularization_losses
@trainable_variables
snon_trainable_variables
Aregularization_losses

tlayers
B	variables
ulayer_metrics
vmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
!0
"1
,2
-3"
trackable_list_wrapper
?
wlayer_regularization_losses
Etrainable_variables
xnon_trainable_variables
Fregularization_losses

ylayers
G	variables
zlayer_metrics
{metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
|layer_regularization_losses
Itrainable_variables
}non_trainable_variables
Jregularization_losses

~layers
K	variables
layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Mtrainable_variables
?non_trainable_variables
Nregularization_losses
?layers
O	variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
%0
&1
.2
/3"
trackable_list_wrapper
?
 ?layer_regularization_losses
Rtrainable_variables
?non_trainable_variables
Sregularization_losses
?layers
T	variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
 ?layer_regularization_losses
Vtrainable_variables
?non_trainable_variables
Wregularization_losses
?layers
X	variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
 ?layer_regularization_losses
Ztrainable_variables
?non_trainable_variables
[regularization_losses
?layers
\	variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
)0
*1
02
13"
trackable_list_wrapper
?
 ?layer_regularization_losses
_trainable_variables
?non_trainable_variables
`regularization_losses
?layers
a	variables
?layer_metrics
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
)__inference_embed_seq_layer_call_fn_90165
)__inference_embed_seq_layer_call_fn_90527
)__inference_embed_seq_layer_call_fn_90570
)__inference_embed_seq_layer_call_fn_90349?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90676
D__inference_embed_seq_layer_call_and_return_conditional_losses_90824
D__inference_embed_seq_layer_call_and_return_conditional_losses_90394
D__inference_embed_seq_layer_call_and_return_conditional_losses_90439?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_88992ipt"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_embed_seq_1_layer_call_fn_89021
+__inference_embed_seq_1_layer_call_fn_90831
+__inference_embed_seq_1_layer_call_fn_90838
+__inference_embed_seq_1_layer_call_fn_89057?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_90850
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_90862
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_89064
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_89071?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_embed_seq_2_layer_call_fn_89781
+__inference_embed_seq_2_layer_call_fn_90899
+__inference_embed_seq_2_layer_call_fn_90936
+__inference_embed_seq_2_layer_call_fn_89971?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_91023
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_91152
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_90015
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_90059?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_conv1d_2_layer_call_fn_91161?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_91176?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_90484ipt"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
#__inference_el1_layer_call_fn_91183?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
>__inference_el1_layer_call_and_return_conditional_losses_91195?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_max_pooling1d_layer_call_fn_91200
-__inference_max_pooling1d_layer_call_fn_91205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_91213
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_91221?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_batch_normalization_layer_call_fn_91234
3__inference_batch_normalization_layer_call_fn_91247?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_91267
N__inference_batch_normalization_layer_call_and_return_conditional_losses_91301?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
&__inference_conv1d_layer_call_fn_91310?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_conv1d_layer_call_and_return_conditional_losses_91326?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling1d_1_layer_call_fn_91331
/__inference_max_pooling1d_1_layer_call_fn_91336?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_91344
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_91352?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_batch_normalization_1_layer_call_fn_91365
5__inference_batch_normalization_1_layer_call_fn_91378?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_91398
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_91432?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_conv1d_1_layer_call_fn_91441?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv1d_1_layer_call_and_return_conditional_losses_91456?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling1d_2_layer_call_fn_91461
/__inference_max_pooling1d_2_layer_call_fn_91466?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_91474
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_91482?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_batch_normalization_2_layer_call_fn_91495
5__inference_batch_normalization_2_layer_call_fn_91508?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_91528
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_91562?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_88992?+-!,"#$/%.&'(1)0*9?6
/?,
*?'
ipt??????????????????
? "@?=
;
conv1d_2/?,
conv1d_2?????????????????? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_91398~/%.&A?>
7?4
.?+
inputs???????????????????
p 
? "3?0
)?&
0???????????????????
? ?
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_91432~./%&A?>
7?4
.?+
inputs???????????????????
p
? "3?0
)?&
0???????????????????
? ?
5__inference_batch_normalization_1_layer_call_fn_91365q/%.&A?>
7?4
.?+
inputs???????????????????
p 
? "&?#????????????????????
5__inference_batch_normalization_1_layer_call_fn_91378q./%&A?>
7?4
.?+
inputs???????????????????
p
? "&?#????????????????????
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_91528|1)0*@?=
6?3
-?*
inputs??????????????????@
p 
? "2?/
(?%
0??????????????????@
? ?
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_91562|01)*@?=
6?3
-?*
inputs??????????????????@
p
? "2?/
(?%
0??????????????????@
? ?
5__inference_batch_normalization_2_layer_call_fn_91495o1)0*@?=
6?3
-?*
inputs??????????????????@
p 
? "%?"??????????????????@?
5__inference_batch_normalization_2_layer_call_fn_91508o01)*@?=
6?3
-?*
inputs??????????????????@
p
? "%?"??????????????????@?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_91267~-!,"A?>
7?4
.?+
inputs???????????????????
p 
? "3?0
)?&
0???????????????????
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_91301~,-!"A?>
7?4
.?+
inputs???????????????????
p
? "3?0
)?&
0???????????????????
? ?
3__inference_batch_normalization_layer_call_fn_91234q-!,"A?>
7?4
.?+
inputs???????????????????
p 
? "&?#????????????????????
3__inference_batch_normalization_layer_call_fn_91247q,-!"A?>
7?4
.?+
inputs???????????????????
p
? "&?#????????????????????
C__inference_conv1d_1_layer_call_and_return_conditional_losses_91456w'(=?:
3?0
.?+
inputs???????????????????
? "2?/
(?%
0??????????????????@
? ?
(__inference_conv1d_1_layer_call_fn_91441j'(=?:
3?0
.?+
inputs???????????????????
? "%?"??????????????????@?
C__inference_conv1d_2_layer_call_and_return_conditional_losses_91176v<?9
2?/
-?*
inputs??????????????????@
? "2?/
(?%
0?????????????????? 
? ?
(__inference_conv1d_2_layer_call_fn_91161i<?9
2?/
-?*
inputs??????????????????@
? "%?"?????????????????? ?
A__inference_conv1d_layer_call_and_return_conditional_losses_91326x#$=?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
&__inference_conv1d_layer_call_fn_91310k#$=?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
>__inference_el1_layer_call_and_return_conditional_losses_91195v+<?9
2?/
-?*
inputs??????????????????
? "3?0
)?&
0???????????????????
? ?
#__inference_el1_layer_call_fn_91183i+<?9
2?/
-?*
inputs??????????????????
? "&?#????????????????????
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_89064?+G?D
=?:
0?-
	el1_input??????????????????
p 

 
? "3?0
)?&
0???????????????????
? ?
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_89071?+G?D
=?:
0?-
	el1_input??????????????????
p

 
? "3?0
)?&
0???????????????????
? ?
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_90850~+D?A
:?7
-?*
inputs??????????????????
p 

 
? "3?0
)?&
0???????????????????
? ?
F__inference_embed_seq_1_layer_call_and_return_conditional_losses_90862~+D?A
:?7
-?*
inputs??????????????????
p

 
? "3?0
)?&
0???????????????????
? ?
+__inference_embed_seq_1_layer_call_fn_89021t+G?D
=?:
0?-
	el1_input??????????????????
p 

 
? "&?#????????????????????
+__inference_embed_seq_1_layer_call_fn_89057t+G?D
=?:
0?-
	el1_input??????????????????
p

 
? "&?#????????????????????
+__inference_embed_seq_1_layer_call_fn_90831q+D?A
:?7
-?*
inputs??????????????????
p 

 
? "&?#????????????????????
+__inference_embed_seq_1_layer_call_fn_90838q+D?A
:?7
-?*
inputs??????????????????
p

 
? "&?#????????????????????
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_90015?-!,"#$/%.&'(1)0*R?O
H?E
;?8
max_pooling1d_input???????????????????
p 

 
? "2?/
(?%
0??????????????????@
? ?
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_90059?,-!"#$./%&'(01)*R?O
H?E
;?8
max_pooling1d_input???????????????????
p

 
? "2?/
(?%
0??????????????????@
? ?
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_91023?-!,"#$/%.&'(1)0*E?B
;?8
.?+
inputs???????????????????
p 

 
? "2?/
(?%
0??????????????????@
? ?
F__inference_embed_seq_2_layer_call_and_return_conditional_losses_91152?,-!"#$./%&'(01)*E?B
;?8
.?+
inputs???????????????????
p

 
? "2?/
(?%
0??????????????????@
? ?
+__inference_embed_seq_2_layer_call_fn_89781?-!,"#$/%.&'(1)0*R?O
H?E
;?8
max_pooling1d_input???????????????????
p 

 
? "%?"??????????????????@?
+__inference_embed_seq_2_layer_call_fn_89971?,-!"#$./%&'(01)*R?O
H?E
;?8
max_pooling1d_input???????????????????
p

 
? "%?"??????????????????@?
+__inference_embed_seq_2_layer_call_fn_90899?-!,"#$/%.&'(1)0*E?B
;?8
.?+
inputs???????????????????
p 

 
? "%?"??????????????????@?
+__inference_embed_seq_2_layer_call_fn_90936?,-!"#$./%&'(01)*E?B
;?8
.?+
inputs???????????????????
p

 
? "%?"??????????????????@?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90394?+-!,"#$/%.&'(1)0*A?>
7?4
*?'
ipt??????????????????
p 

 
? "2?/
(?%
0?????????????????? 
? ?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90439?+,-!"#$./%&'(01)*A?>
7?4
*?'
ipt??????????????????
p

 
? "2?/
(?%
0?????????????????? 
? ?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90676?+-!,"#$/%.&'(1)0*D?A
:?7
-?*
inputs??????????????????
p 

 
? "2?/
(?%
0?????????????????? 
? ?
D__inference_embed_seq_layer_call_and_return_conditional_losses_90824?+,-!"#$./%&'(01)*D?A
:?7
-?*
inputs??????????????????
p

 
? "2?/
(?%
0?????????????????? 
? ?
)__inference_embed_seq_layer_call_fn_90165+-!,"#$/%.&'(1)0*A?>
7?4
*?'
ipt??????????????????
p 

 
? "%?"?????????????????? ?
)__inference_embed_seq_layer_call_fn_90349+,-!"#$./%&'(01)*A?>
7?4
*?'
ipt??????????????????
p

 
? "%?"?????????????????? ?
)__inference_embed_seq_layer_call_fn_90527?+-!,"#$/%.&'(1)0*D?A
:?7
-?*
inputs??????????????????
p 

 
? "%?"?????????????????? ?
)__inference_embed_seq_layer_call_fn_90570?+,-!"#$./%&'(01)*D?A
:?7
-?*
inputs??????????????????
p

 
? "%?"?????????????????? ?
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_91344?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_91352t=?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
/__inference_max_pooling1d_1_layer_call_fn_91331wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
/__inference_max_pooling1d_1_layer_call_fn_91336g=?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_91474?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_91482r<?9
2?/
-?*
inputs??????????????????@
? "2?/
(?%
0??????????????????@
? ?
/__inference_max_pooling1d_2_layer_call_fn_91461wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
/__inference_max_pooling1d_2_layer_call_fn_91466e<?9
2?/
-?*
inputs??????????????????@
? "%?"??????????????????@?
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_91213?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_91221t=?:
3?0
.?+
inputs???????????????????
? "3?0
)?&
0???????????????????
? ?
-__inference_max_pooling1d_layer_call_fn_91200wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
-__inference_max_pooling1d_layer_call_fn_91205g=?:
3?0
.?+
inputs???????????????????
? "&?#????????????????????
#__inference_signature_wrapper_90484?+-!,"#$/%.&'(1)0*@?=
? 
6?3
1
ipt*?'
ipt??????????????????"@?=
;
conv1d_2/?,
conv1d_2?????????????????? 