import pytest

from realutils.tagging import get_idolsankaku_tags, convert_idolsankaku_emb_to_prediction
from realutils.tagging.idolsankaku import _get_idolsankaku_labels, _get_idolsankaku_model
from test.testings import get_testfile


@pytest.fixture(scope='module', autouse=True)
def _release_model():
    try:
        yield
    finally:
        _get_idolsankaku_labels.cache_clear()
        _get_idolsankaku_model.cache_clear()


@pytest.fixture()
def input_image_1():
    return get_testfile('idolsankaku', '1.jpg')


@pytest.fixture()
def expected_result_1():
    rating = {'explicit': 0.022273868322372437,
              'questionable': 0.22442740201950073,
              'safe': 0.748395562171936}
    general = {'1girl': 0.7476911544799805,
               'asian': 0.3681548237800598,
               'blouse': 0.7909733057022095,
               'brown_hair': 0.4968719780445099,
               'high_heels': 0.41397374868392944,
               'long_hair': 0.7415428161621094,
               'non_nude': 0.4075928330421448,
               'outdoors': 0.5279690623283386,
               'pantyhose': 0.8893758654594421,
               'sitting': 0.49351146817207336,
               'skirt': 0.8094233274459839,
               'solo': 0.44033104181289673}
    character = {}
    return rating, general, character


@pytest.fixture()
def input_image_2():
    return get_testfile('idolsankaku', '2.jpg')


@pytest.fixture()
def expected_result_2():
    rating = {'explicit': 0.0979660153388977,
              'questionable': 0.6467134952545166,
              'safe': 0.19094136357307434}
    general = {'1girl': 0.8127931356430054,
               'asian': 0.9137638807296753,
               'bed': 0.4242570698261261,
               'black_hair': 0.7014122605323792,
               'female': 0.4146701693534851,
               'female_only': 0.4589616656303406,
               'open_clothes': 0.626146137714386,
               'pleated_skirt': 0.45050889253616333,
               'school_uniform': 0.7465487718582153,
               'sitting': 0.3599088788032532,
               'skirt': 0.8968819975852966,
               'solo': 0.7613500356674194}
    character = {}
    return rating, general, character


@pytest.fixture()
def input_image_3():
    return get_testfile('idolsankaku', '3.jpg')


@pytest.fixture()
def expected_result_3():
    rating = {'explicit': 0.012922674417495728,
              'questionable': 0.5239537358283997,
              'safe': 0.2681353688240051}
    general = {'1girl': 0.7474421858787537,
               'asian': 0.8060781955718994,
               'bikini': 0.8324492573738098,
               'breasts': 0.8964301347732544,
               'brown_eyes': 0.5026739239692688,
               'brown_hair': 0.5091124176979065,
               'japanese': 0.4774506688117981,
               'large_breasts': 0.8372409343719482,
               'long_hair': 0.4838402271270752,
               'looking_at_viewer': 0.7040004134178162,
               'navel': 0.7938550710678101,
               'sitting': 0.6388136148452759,
               'solo': 0.4194791316986084,
               'swimsuit': 0.3804645240306854,
               'underboob': 0.5410750508308411}
    character = {}
    return rating, general, character


@pytest.fixture()
def input_image_4():
    return get_testfile('idolsankaku', '4.jpg')


@pytest.fixture()
def expected_result_4():
    rating = {'explicit': 0.014742940664291382,
              'questionable': 0.8283720016479492,
              'safe': 0.16998109221458435}
    general = {'1girl': 0.7149745225906372,
               'animal_ears': 0.7749373912811279,
               'asian': 0.8063126802444458,
               'bikini': 0.8782480359077454,
               'breasts': 0.7182254791259766,
               'brown_hair': 0.5295036435127258,
               'east_asian': 0.7766237258911133,
               'female': 0.49066367745399475,
               'japanese': 0.9429196119308472,
               'large_breasts': 0.3805162310600281,
               'long_hair': 0.4580710232257843,
               'looking_at_viewer': 0.7904833555221558,
               'smile': 0.7558436393737793,
               'solo': 0.5720633864402771,
               'swimsuit': 0.4878530502319336}
    character = {}
    return rating, general, character


@pytest.fixture()
def input_image_5():
    return get_testfile('idolsankaku', '5.jpg')


@pytest.fixture()
def expected_result_5():
    rating = {'explicit': 0.9595806002616882,
              'questionable': 0.01747998595237732,
              'safe': 0.037566035985946655}
    general = {'back': 0.7824065089225769,
               'breasts': 0.9307494163513184,
               'brown_hair': 0.5315978527069092,
               'high_heels': 0.9425466060638428,
               'mirror': 0.9521722793579102,
               'nipples': 0.9344090223312378,
               'oshiri': 0.8306560516357422,
               'pantsu': 0.9224182367324829,
               'reflection': 0.7433485388755798,
               'squat': 0.7803635001182556,
               'thighhighs': 0.8777117729187012,
               'thong': 0.8730485439300537,
               'topless': 0.6496016979217529}
    character = {}
    return rating, general, character


@pytest.fixture()
def input_image_6():
    return get_testfile('idolsankaku', '6.jpg')


@pytest.fixture()
def expected_result_6():
    rating = {'explicit': 0.9775880575180054,
              'questionable': 0.010381460189819336,
              'safe': 0.01930221915245056}
    general = {'bed': 0.6602951288223267,
               'breasts': 0.9259153008460999,
               'brown_hair': 0.7530864477157593,
               'cleavage': 0.37149950861930847,
               'covering_pussy': 0.49498632550239563,
               'kneeling': 0.8785718679428101,
               'navel': 0.8053578734397888,
               'nipples': 0.9164817333221436,
               'nude': 0.9233779907226562,
               'pubic_hair': 0.8027235269546509}
    character = {}
    return rating, general, character


@pytest.fixture()
def input_image_7():
    return get_testfile('idolsankaku', '7.jpg')


@pytest.fixture()
def expected_result_7():
    rating = {'explicit': 0.0018109679222106934,
              'questionable': 0.0257779061794281,
              'safe': 0.9750080704689026}
    general = {'1girl': 0.5759814381599426,
               'aqua_hair': 0.9376567006111145,
               'armpit': 0.5968506336212158,
               'arms_up': 0.9492673873901367,
               'asian': 0.46296364068984985,
               'black_thighhighs': 0.41496211290359497,
               'blouse': 0.8670071959495544,
               'default_costume': 0.36392033100128174,
               'detached_sleeves': 0.9382797479629517,
               'female': 0.5258357524871826,
               'long_hair': 0.8752110004425049,
               'looking_at_viewer': 0.4927205741405487,
               'miniskirt': 0.8354354500770569,
               'pleated_skirt': 0.8233045935630798,
               'shirt': 0.8463951945304871,
               'skirt': 0.9698911905288696,
               'sleeveless': 0.9865490198135376,
               'sleeveless_blouse': 0.9789504408836365,
               'sleeveless_shirt': 0.9865082502365112,
               'solo': 0.6263223886489868,
               'tie': 0.8901710510253906,
               'twintails': 0.9444552659988403,
               'very_long_hair': 0.3988983631134033}
    character = {'hatsune_miku': 0.9460012912750244}
    return rating, general, character


@pytest.fixture()
def input_image_8():
    return get_testfile('idolsankaku', '8.jpg')


@pytest.fixture()
def expected_result_8():
    rating = {'explicit': 0.9970412254333496,
              'questionable': 0.0028026998043060303,
              'safe': 0.004712015390396118}
    general = {'1girl': 0.9889324903488159,
               'aqua_eyes': 0.8344452977180481,
               'aqua_hair': 0.8597970008850098,
               'asian': 0.7511153221130371,
               'black_legwear': 0.4415139853954315,
               'breasts': 0.4875076115131378,
               'detached_sleeves': 0.7122460007667542,
               'female': 0.8567990660667419,
               'female_only': 0.7391462922096252,
               'female_solo': 0.7551952600479126,
               'nipples': 0.6325450539588928,
               'nude': 0.6523562669754028,
               'solo': 0.9733173847198486,
               'twintails': 0.7836702466011047,
               'vagina': 0.6575908064842224}
    character = {'hatsune_miku': 0.9384632110595703}
    return rating, general, character


@pytest.fixture()
def all_fixes(input_image_1, expected_result_1, input_image_2, expected_result_2,
              input_image_3, expected_result_3, input_image_4, expected_result_4,
              input_image_5, expected_result_5, input_image_6, expected_result_6,
              input_image_7, expected_result_7, input_image_8, expected_result_8):
    return {1: (input_image_1, expected_result_1),
            2: (input_image_2, expected_result_2),
            3: (input_image_3, expected_result_3),
            4: (input_image_4, expected_result_4),
            5: (input_image_5, expected_result_5),
            6: (input_image_6, expected_result_6),
            7: (input_image_7, expected_result_7),
            8: (input_image_8, expected_result_8)}


@pytest.fixture()
def expected_result_no_underline_1():
    rating = {'explicit': 0.022273868322372437,
              'questionable': 0.22442740201950073,
              'safe': 0.748395562171936}
    general = {'1girl': 0.7476911544799805,
               'asian': 0.3681548237800598,
               'blouse': 0.7909733057022095,
               'brown hair': 0.4968719780445099,
               'high heels': 0.41397374868392944,
               'long hair': 0.7415428161621094,
               'non nude': 0.4075928330421448,
               'outdoors': 0.5279690623283386,
               'pantyhose': 0.8893758654594421,
               'sitting': 0.49351146817207336,
               'skirt': 0.8094233274459839,
               'solo': 0.44033104181289673}
    character = {}
    return rating, general, character


@pytest.fixture()
def expected_result_no_underline_8():
    rating = {'explicit': 0.9970412254333496,
              'questionable': 0.0028026998043060303,
              'safe': 0.004712015390396118}
    general = {'1girl': 0.9889324903488159,
               'aqua eyes': 0.8344452977180481,
               'aqua hair': 0.8597970008850098,
               'asian': 0.7511153221130371,
               'black legwear': 0.4415139853954315,
               'breasts': 0.4875076115131378,
               'detached sleeves': 0.7122460007667542,
               'female': 0.8567990660667419,
               'female only': 0.7391462922096252,
               'female solo': 0.7551952600479126,
               'nipples': 0.6325450539588928,
               'nude': 0.6523562669754028,
               'solo': 0.9733173847198486,
               'twintails': 0.7836702466011047,
               'vagina': 0.6575908064842224}
    character = {'hatsune miku': 0.9384632110595703}
    return rating, general, character


@pytest.fixture()
def all_no_underline_fixes(input_image_1, expected_result_no_underline_1,
                           input_image_8, expected_result_no_underline_8):
    return {1: (input_image_1, expected_result_no_underline_1),
            8: (input_image_8, expected_result_no_underline_8)}


@pytest.fixture()
def expected_result_mcut_1():
    rating = {'explicit': 0.022273868322372437,
              'questionable': 0.22442740201950073,
              'safe': 0.748395562171936}
    general = {'1girl': 0.7476911544799805,
               'blouse': 0.7909733057022095,
               'long_hair': 0.7415428161621094,
               'pantyhose': 0.8893758654594421,
               'skirt': 0.8094233274459839}
    character = {}
    return rating, general, character


@pytest.fixture()
def expected_result_mcut_2():
    rating = {'explicit': 0.0979660153388977,
              'questionable': 0.6467134952545166,
              'safe': 0.19094136357307434}
    general = {'1girl': 0.8127931356430054,
               'asian': 0.9137638807296753,
               'black_hair': 0.7014122605323792,
               'open_clothes': 0.626146137714386,
               'school_uniform': 0.7465487718582153,
               'skirt': 0.8968819975852966,
               'solo': 0.7613500356674194}
    character = {}
    return rating, general, character


@pytest.fixture()
def expected_result_mcut_3():
    rating = {'explicit': 0.012922674417495728,
              'questionable': 0.5239537358283997,
              'safe': 0.2681353688240051}
    general = {'1girl': 0.7474421858787537,
               'asian': 0.8060781955718994,
               'bikini': 0.8324492573738098,
               'breasts': 0.8964301347732544,
               'large_breasts': 0.8372409343719482,
               'looking_at_viewer': 0.7040004134178162,
               'navel': 0.7938550710678101,
               'sitting': 0.6388136148452759}
    character = {}
    return rating, general, character


@pytest.fixture()
def expected_result_mcut_4():
    rating = {'explicit': 0.014742940664291382,
              'questionable': 0.8283720016479492,
              'safe': 0.16998109221458435}
    general = {'1girl': 0.7149745225906372,
               'animal_ears': 0.7749373912811279,
               'asian': 0.8063126802444458,
               'bikini': 0.8782480359077454,
               'breasts': 0.7182254791259766,
               'east_asian': 0.7766237258911133,
               'japanese': 0.9429196119308472,
               'looking_at_viewer': 0.7904833555221558,
               'smile': 0.7558436393737793}
    character = {}
    return rating, general, character


@pytest.fixture()
def expected_result_mcut_5():
    rating = {'explicit': 0.9595806002616882,
              'questionable': 0.01747998595237732,
              'safe': 0.037566035985946655}
    general = {'back': 0.7824065089225769,
               'breasts': 0.9307494163513184,
               'brown_hair': 0.5315978527069092,
               'high_heels': 0.9425466060638428,
               'mirror': 0.9521722793579102,
               'nipples': 0.9344090223312378,
               'oshiri': 0.8306560516357422,
               'pantsu': 0.9224182367324829,
               'reflection': 0.7433485388755798,
               'squat': 0.7803635001182556,
               'thighhighs': 0.8777117729187012,
               'thong': 0.8730485439300537,
               'topless': 0.6496016979217529}
    character = {}
    return rating, general, character


@pytest.fixture()
def expected_result_mcut_6():
    rating = {'explicit': 0.9775880575180054,
              'questionable': 0.010381460189819336,
              'safe': 0.01930221915245056}
    general = {'bed': 0.6602951288223267,
               'breasts': 0.9259153008460999,
               'brown_hair': 0.7530864477157593,
               'kneeling': 0.8785718679428101,
               'navel': 0.8053578734397888,
               'nipples': 0.9164817333221436,
               'nude': 0.9233779907226562,
               'pubic_hair': 0.8027235269546509}
    character = {}
    return rating, general, character


@pytest.fixture()
def expected_result_mcut_7():
    rating = {'explicit': 0.0018109679222106934,
              'questionable': 0.0257779061794281,
              'safe': 0.9750080704689026}
    general = {'aqua_hair': 0.9376567006111145,
               'arms_up': 0.9492673873901367,
               'blouse': 0.8670071959495544,
               'detached_sleeves': 0.9382797479629517,
               'long_hair': 0.8752110004425049,
               'miniskirt': 0.8354354500770569,
               'pleated_skirt': 0.8233045935630798,
               'shirt': 0.8463951945304871,
               'skirt': 0.9698911905288696,
               'sleeveless': 0.9865490198135376,
               'sleeveless_blouse': 0.9789504408836365,
               'sleeveless_shirt': 0.9865082502365112,
               'tie': 0.8901710510253906,
               'twintails': 0.9444552659988403}
    character = {'hatsune_miku': 0.9460012912750244}
    return rating, general, character


@pytest.fixture()
def expected_result_mcut_8():
    rating = {'explicit': 0.9970412254333496,
              'questionable': 0.0028026998043060303,
              'safe': 0.004712015390396118}
    general = {'1girl': 0.9889324903488159,
               'aqua_eyes': 0.8344452977180481,
               'aqua_hair': 0.8597970008850098,
               'asian': 0.7511153221130371,
               'black_legwear': 0.4415139853954315,
               'breasts': 0.4875076115131378,
               'detached_sleeves': 0.7122460007667542,
               'female': 0.8567990660667419,
               'female_only': 0.7391462922096252,
               'female_solo': 0.7551952600479126,
               'long_hair': 0.33724209666252136,
               'looking_at_viewer': 0.27091294527053833,
               'nipples': 0.6325450539588928,
               'nude': 0.6523562669754028,
               'solo': 0.9733173847198486,
               'twintails': 0.7836702466011047,
               'vagina': 0.6575908064842224}
    character = {'hatsune_miku': 0.9384632110595703}
    return rating, general, character


@pytest.fixture()
def all_mcut_fixes(input_image_1, expected_result_mcut_1, input_image_2, expected_result_mcut_2, input_image_3,
                   expected_result_mcut_3, input_image_4, expected_result_mcut_4, input_image_5, expected_result_mcut_5,
                   input_image_6, expected_result_mcut_6, input_image_7, expected_result_mcut_7, input_image_8,
                   expected_result_mcut_8):
    return {1: (input_image_1, expected_result_mcut_1),
            2: (input_image_2, expected_result_mcut_2),
            3: (input_image_3, expected_result_mcut_3),
            4: (input_image_4, expected_result_mcut_4),
            5: (input_image_5, expected_result_mcut_5),
            6: (input_image_6, expected_result_mcut_6),
            7: (input_image_7, expected_result_mcut_7),
            8: (input_image_8, expected_result_mcut_8)}


@pytest.mark.unittest
class TestTaggingIdolsankaku:
    @pytest.mark.parametrize(['sample_id'], [(sid,) for sid in range(1, 9)])
    def test_get_idolsankaku_tags(self, sample_id, all_fixes):
        input_, (expected_rating, expected_general, expected_character) = all_fixes[sample_id]
        rating, general, character = get_idolsankaku_tags(input_)
        assert rating == pytest.approx(expected_rating, abs=1e-3)
        assert general == pytest.approx(expected_general, abs=1e-3)
        assert character == pytest.approx(expected_character, abs=1e-3)

    @pytest.mark.parametrize(['sample_id'], [(sid,) for sid in [1, 8]])
    def test_get_idolsankaku_tags_no_underline(self, sample_id, all_no_underline_fixes):
        input_, (expected_rating, expected_general, expected_character) = all_no_underline_fixes[sample_id]
        rating, general, character = get_idolsankaku_tags(input_, no_underline=True)
        assert rating == pytest.approx(expected_rating, abs=1e-3)
        assert general == pytest.approx(expected_general, abs=1e-3)
        assert character == pytest.approx(expected_character, abs=1e-3)

    @pytest.mark.parametrize(['sample_id'], [(sid,) for sid in range(1, 9)])
    def test_get_idolsankaku_tags_mcut(self, sample_id, all_mcut_fixes):
        input_, (expected_rating, expected_general, expected_character) = all_mcut_fixes[sample_id]
        rating, general, character = \
            get_idolsankaku_tags(input_, character_mcut_enabled=True, general_mcut_enabled=True)
        assert rating == pytest.approx(expected_rating, abs=1e-3)
        assert general == pytest.approx(expected_general, abs=1e-3)
        assert character == pytest.approx(expected_character, abs=1e-3)

    @pytest.mark.parametrize(['sample_id'], [(sid,) for sid in range(1, 9)])
    def test_convert_idolsankaku_emb_to_prediction(self, sample_id, all_fixes):
        input_, _ = all_fixes[sample_id]
        expected_rating, expected_general, expected_character, embedding = get_idolsankaku_tags(
            input_, fmt=('rating', 'general', 'character', 'embedding'))
        rating, general, character = convert_idolsankaku_emb_to_prediction(embedding)
        assert rating == pytest.approx(expected_rating, abs=1e-3)
        assert general == pytest.approx(expected_general, abs=1e-3)
        assert character == pytest.approx(expected_character, abs=1e-3)

    @pytest.mark.parametrize(['sample_id'], [(sid,) for sid in range(1, 9)])
    def test_convert_idolsankaku_emb_to_prediction_lst(self, sample_id, all_fixes):
        input_, _ = all_fixes[sample_id]
        expected_rating, expected_general, expected_character, embedding = get_idolsankaku_tags(
            input_, fmt=('rating', 'general', 'character', 'embedding'))
        result = convert_idolsankaku_emb_to_prediction(embedding[None, ...])
        assert isinstance(result, list)
        assert len(result) == 1
        rating, general, character = result[0]
        assert rating == pytest.approx(expected_rating, abs=1e-3)
        assert general == pytest.approx(expected_general, abs=1e-3)
        assert character == pytest.approx(expected_character, abs=1e-3)
