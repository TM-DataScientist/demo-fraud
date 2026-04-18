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


"""学習済み分類器を推論 API として提供するサービング実装。"""

import numpy as np
from cloudpickle import load
from mlrun.serving.v2_serving import V2ModelServer


class ClassifierModel(V2ModelServer):
    """
    Model serving classifer example
    学習済み分類モデルを読み込み、推論エンドポイントとして公開するクラス。
    """

    def load(self):
        """
        load and initialize the model and/or other elements
        モデルアーティファクトを取得して、推論に使える状態へ初期化する。
        """
        model_file, extra_data = self.get_model(".pkl")
        # 取得した pickle ファイルを復元して、リクエストごとに再利用できるよう保持する。
        self.model = load(open(model_file, "rb"))

    def predict(self, body: dict) -> list:
        """
        Generate model predictions from sample
        入力リクエストから特徴量配列を取り出し、分類結果を返す。
        """
        # V2 Serving のリクエスト本文から inputs を取り出して NumPy 配列へ変換する。
        print(f"Input -> {body['inputs']}")
        feats = np.asarray(body["inputs"])
        # 学習済みモデルでバッチ推論を行い、JSON で返しやすい list に変換する。
        result: np.ndarray = self.model.predict(feats)
        return result.tolist()
