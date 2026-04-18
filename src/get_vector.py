# Copyright 2024 Iguazio
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
#
"""Feature Store から学習用のオフライン特徴量を取得する処理。"""

import mlrun.feature_store as fstore
from mlrun.datastore.targets import ParquetTarget


def get_offline_features(feature_vector, features, label_feature):
    """
    Build a feature vector definition and materialize it to an offline target.
    特徴量ベクトルを定義し、学習や検証に使えるオフラインデータとして取得する。

    :param feature_vector: FeatureVector 名または URI
    :param features: 取得対象の特徴量一覧
    :param label_feature: ラベル列として扱う特徴量
    :returns: 取得済みオフライン特徴量オブジェクト
    """
    # FeatureVector の定義をその場で構築し、後続の取得処理に渡す。
    fv = fstore.FeatureVector(
        feature_vector,
        features,
        label_feature=label_feature,
        description="Predicting a fraudulent transaction",
    )

    # Parquet へ書き出せるターゲットを指定して、オフライン特徴量を実体化する。
    data = fv.get_offline_features(target=ParquetTarget())
    return data
