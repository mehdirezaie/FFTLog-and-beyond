
. "/Users/rezaie/anaconda3/etc/profile.d/conda.sh"
export NUMEXPR_MAX_THREADS=2
export PYTHONPATH=${HOME}/github/LSSutils:${HOME}/github/sysnetdev
conda activate sysnet




nside=512
bsize=1000
regions='bass'
axes=({0..20}) # 'nstar', 'ebv', 'loghi', 'ccdskymag_g_mean', 'fwhm_g_mean', 'depth_g_total', 'mjd_g_min', airmass, exptime
nns=(5 10)
lr=0.2
etmin=0.0001
nepoch=300
nchain=20

input_dir=/Volumes/TimeMachine/data/DR9fnl/qso/regression/bass/
input_path=${input_dir}nqso_bass_${nside}.fits

do_lr=false
do_nnfit=true



nnfit=${HOME}/github/sysnetdev/scripts/app.py


if [ "${do_lr}" = true ]
then
    for region in ${regions}
    do
        du -h ${input_path}

        output_path=${input_dir}nn_mse/hp/
        
        echo ${output_path}
        python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model dnn --loss mse --nn_structure ${nns[@]} -fl
    done
fi


if [ "${do_nnfit}" = true ]
then
    for region in ${regions}
    do
        du -h ${input_path}
        output_path=${input_dir}nn_mse/
        echo ${output_path}
        python $nnfit -i ${input_path} -o ${output_path} -ax ${axes[@]} -bs ${bsize} --model dnn --loss mse -lr $lr --eta_min $etmin -ne $nepoch --nn_structure ${nns[@]} -k -nc $nchain
    done
fi
