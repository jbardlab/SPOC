container="spoc:v1"
repo_dir="/Users/jbard/Library/CloudStorage/OneDrive-TexasA&MUniversity/repos/SPOC"
input_dir="/Users/jbard/Library/CloudStorage/OneDrive-TexasA&MUniversity/repos/SPOC/example/spoctest_GNAS2_GP119_v3"

docker run --rm \
  -v "${input_dir}":/input \
  -v "${repo_dir}":/repo \
  ${container} \
  bash -c "python /repo/run_custom.py /input"