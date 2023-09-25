from classification_model.config.core import config
from classification_model.processing.features import ExtractLetterTransformer


def test_extract_letter_transformer(sample_input_data):
    # Given
    transformer = ExtractLetterTransformer(
        variables=config.model_config.cabin_vars
    )

    sample_input_data.to_csv("fuck_this.csv")
    assert sample_input_data["cabin"].iat[9] == 'A26'

    # When
    subject = transformer.fit_transform(sample_input_data)

    # Then
    assert subject["cabin"].iat[9] == 'A'
