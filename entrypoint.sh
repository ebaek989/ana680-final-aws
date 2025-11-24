set -e

mode="$1"
if [[ "$mode" == "train" ]]; then
  shift
  echo "Running training..."
  exec python train.py "$@"
elif [[ "$mode" == "serve" ]]; then
  shift
  echo "Starting inference server..."
  exec ./serve "$@"
else
  echo "No mode specified, defaulting to 'serve'..."
  exec ./serve "$@"
fi
