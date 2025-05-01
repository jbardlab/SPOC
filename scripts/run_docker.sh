container="spoc:v1"
repo_dir="/Users/jbard/Library/CloudStorage/OneDrive-TexasA&MUniversity/repos/SPOC"
input_dir="/Users/jbard/Library/CloudStorage/OneDrive-TexasA&MUniversity/repos/SPOC/example/spoctest_GNAS2_GP119_v3"

docker run --rm \
  -v "${input_dir}":/input \
  -v "${repo_dir}":/repo \
  ${container} \
  bash -c \
  "python /repo/scripts/run_custom_nobio.py /input --rf_params /repo/models/rf_afm_no_bio.joblib --output /input/spoc_nobio_output.csv --ipsae_script /repo/scripts/ipsae.py"