sh '''pwd
#!/bin/bash

# Install packages from requirements.txt with \'python3-\' prefix

# Read packages from requirements.txt and install each with \'python3-\'
while read -r line; do
  package=$(echo "$line" | awk \'{print $1}\')
  if [[ $package != "python-dateutil" ]]; then
    package="python3-${package}"
  fi
  version=$(echo "$line" | awk \'{print $3}\')
  if [[ -z "$version" ]]; then
    # If no version specified, install the latest version
    pip install "$package"
  else
    pip install "$package==$version"
  fi
done < requirements.txt

'''