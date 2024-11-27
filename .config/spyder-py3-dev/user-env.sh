!/bin/bash -i
unset HISTFILE
/bin/bash -l -c "'/usr/bin/python3' -c 'import os; print(dict(os.environ))'"
