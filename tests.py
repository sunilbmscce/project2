import modeling_complete


def test_loaddata_dimensions():
    X, Y  = modeling_complete.load_hospital_data()
    assert X.shape[0] == Y.shape[0]
