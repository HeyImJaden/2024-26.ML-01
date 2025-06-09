import pytest
from app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_hello(client):
    response = client.post("/infer", 
                           json={
                                "features": [3058,2,1,2017,1.498552073529678,0,2,672852.0519810001],
                                })
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"message": "Hello Stranger!"}