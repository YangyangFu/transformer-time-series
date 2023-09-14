# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
gdown --fuzzy https://drive.google.com/file/d/1h2ubecIXd6Ji_mtEKin5G0-KtqNH09T5/view?usp=drive_link
unzip time-series-datasets.zip
mv time-series-datasets/* .
rm -rf time-series-datasets*