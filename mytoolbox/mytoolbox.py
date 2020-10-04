import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class PreProcessingTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Xtemp = X.copy()

        # Height
        index_height = Xtemp.loc[Xtemp['height'] > 230, ['height']].index
        Xtemp.drop(index_height, inplace=True)
        index_height1 = Xtemp.loc[Xtemp['height'] < 112, ['height']].index
        Xtemp.drop(index_height1, inplace=True)

        # Weight
        index_weight = Xtemp.loc[Xtemp['weight'] < 40, ['weight']].index
        Xtemp.drop(index_weight, inplace=True)

        # ap_hi
        index_ap_hi = Xtemp.loc[Xtemp['ap_hi'] < 10, ['ap_hi']].index
        Xtemp.drop(index_ap_hi, inplace=True)

        # ap_lo
        index_ap_lo = Xtemp.loc[Xtemp['ap_lo'] < 5, ['ap_lo']].index
        Xtemp.drop(index_ap_lo, inplace=True)

        return Xtemp


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Xtemp = X.copy()

        # drop 'id' and 'age' 'smoke','alco','gluc', 'ap_lo', 'cholesterol', 'height', 'active', 'weight'
        Xtemp.drop('id', inplace=True, axis=1)

        # IMC
        Xtemp['imc'] = Xtemp['weight']/(Xtemp['height']/100)**2

        # cat_dwarfism
        Xtemp['cat_Dwarfism'] = [1 if value < 145 else 0 for value in Xtemp['height']]

        # ap_hi divide 10
        Xtemp.loc[Xtemp['ap_hi'] > 220, ['ap_hi']] = Xtemp.loc[Xtemp['ap_hi'] > 220, ['ap_hi']]/10

        # ap_lo divide 10
        Xtemp.loc[Xtemp['ap_lo'] > 190, ['ap_lo']] = Xtemp.loc[Xtemp['ap_lo'] > 190, ['ap_lo']]/10

        # ap_hi divide 10
        Xtemp.loc[Xtemp['ap_hi'] > 220, ['ap_hi']] = Xtemp.loc[Xtemp['ap_hi'] > 220,['ap_hi']]/10

        # ap_hi divide 10
        Xtemp.loc[Xtemp['ap_hi'] > 220, ['ap_hi']] = Xtemp.loc[Xtemp['ap_hi'] > 220,['ap_hi']]/10

        # ap_lo divide 10
        Xtemp.loc[Xtemp['ap_lo'] > 190, ['ap_lo']] = Xtemp.loc[Xtemp['ap_lo'] > 190,['ap_lo']]/10


        return Xtemp


class CatBloodPressureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Xtemp = X.copy()

        # cat_bloodpressure
        def cat_bloodpressure(df):

            if df['ap_hi'] < 90 and df['ap_lo'] < 60:
                return 1 # Hipotensão
            elif 90 <= df['ap_hi'] < 140 and 60 <= df['ap_lo'] < 90:
                return 2    # Pré-Hipotensão
            elif 140 <= df['ap_hi'] < 160 and 90 <= df['ap_lo'] < 100:
                return 3  # 'Hipertensão estagio1'
            elif df['ap_hi'] >= 160 and df['ap_lo'] >= 100:
                return 4 # 'Hipertensão estagio2'
            else:
                return 5 # 'no_cat'

        # cat_bloodpressure
        Xtemp['cat_bloodpressure'] = Xtemp.apply(cat_bloodpressure, axis=1)

        return Xtemp


class TotalPressureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        Xtemp = X.copy()

        # total_preassure
        Xtemp['total_pressure'] = Xtemp['ap_hi'] + Xtemp['ap_lo']

        return Xtemp


class MyRobustScalerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        Xtemp = X.copy()

        scaler = RobustScaler()
        Xscaled = scaler.fit_transform(Xtemp)
        Xtemp = pd.DataFrame(Xscaled, columns=Xtemp.columns.to_list())

        return Xtemp


class MyOneHotEncoderTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        Xtemp = X.copy()

        scaler = OneHotEncoder(drop='first')
        Xscaled = scaler.fit_transform(Xtemp)
        colunas = list(scaler.get_feature_names(['cholesterol', 'gluc', 'active', 'gender', 'smoke', 'alco']))
        Xtemp = pd.DataFrame(Xscaled.toarray(), columns=colunas)

        return Xtemp
