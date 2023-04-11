from flow360.component.resource_base import Flow360Status, Flow360ResourceBaseModel
from flow360.component.case import CaseMeta


def test_status():
    assert Flow360Status("error").is_final()
    assert Flow360Status("uploaded").is_final()
    assert Flow360Status("processed").is_final()
    assert Flow360Status("diverged").is_final()

    assert Flow360Status.ERROR.is_final()
    assert Flow360Status.UPLOADED.is_final()
    assert Flow360Status.PROCESSED.is_final()
    assert Flow360Status.DIVERGED.is_final()

    assert not Flow360Status.PREPROCESSING.is_final()
    assert not Flow360Status.RUNNING.is_final()
    assert not Flow360Status.UPLOADING.is_final()
    assert not Flow360Status.GENERATING.is_final()
    assert not Flow360Status.STOPPED.is_final()


def test_base_model():
    model = Flow360ResourceBaseModel(
        status="completed", name="name", userId="userId", deleted=True, id="0"
    )
    assert model.status.is_final()

    model = Flow360ResourceBaseModel(
        status="running", name="name", userId="userId", deleted=True, id="0"
    )
    assert not model.status.is_final()

    model = Flow360ResourceBaseModel.parse_obj(
        {"status": "completed", "name": "name", "userId": "userId", "deleted": True, "id": "0"}
    )
    assert model.status.is_final()

    model = Flow360ResourceBaseModel.parse_obj(
        {"status": "running", "name": "name", "userId": "userId", "deleted": True, "id": "0"}
    )
    assert not model.status.is_final()


def test_case_meta_model():
    model = CaseMeta(
        status="completed",
        name="name",
        userId="userId",
        deleted=True,
        caseId="caseID",
        caseMeshId="caseMeshId",
        parentId="None",
    )
    assert model.status.is_final()

    model = CaseMeta(
        status="running",
        name="name",
        userId="userId",
        deleted=True,
        caseId="caseID",
        caseMeshId="caseMeshId",
        parentId="None",
    )
    assert not model.status.is_final()

    model = CaseMeta.parse_obj(
        {
            "status": "completed",
            "name": "name",
            "userId": "userId",
            "deleted": True,
            "caseId": "caseID",
            "caseMeshId": "caseMeshId",
            "parentId": "None",
        }
    )
    assert model.status.is_final()

    model = CaseMeta.parse_obj(
        {
            "status": "running",
            "name": "name",
            "userId": "userId",
            "deleted": True,
            "caseId": "caseID",
            "caseMeshId": "caseMeshId",
            "parentId": "None",
        }
    )
    assert not model.status.is_final()

    resp = {
        "caseId": "e07abd8b-cc30-4fd8-9159-c7654fe32ca8",
        "caseMeshId": "a52aa1f9-47f6-4041-9f6d-350389efe315",
        "caseName": "case-fork",
        "casePriority": None,
        "caseStartTime": "2023-03-07T14:44:41.375Z",
        "caseSubmitTime": "2023-03-07T14:43:22.036Z",
        "caseFinishTime": "2023-03-07T14:45:25.828Z",
        "caseParentId": "611ef260-84ac-49c9-8de5-8e48062a7763",
        "userId": "AIDAXWCOWJGJINZTEEEIP",
        "caseTags": ["None"],
        "meshSize": 13560436,
        "nodesInfo": None,
        "solverVersion": "release-23.1.1.0",
        "worker": "B.XS.node5",
        "userEmail": "maciej@flexcompute.com",
        "estimationDuration": 0.0,
        "meshNodeSize": 113945,
        "realFlexUnit": 0.027972380392156868,
        "workerGroup": None,
        "estWorkUnit": 0.027972380392156868,
        "confidential": False,
        "storageSize": 67511271,
        "storageStatus": "STANDARD",
        "standardAge": 1,
        "storageClass": "STANDARD",
        "restoreAt": None,
        "deleted": False,
        "errorType": None,
        "numProcessors": "6",
        "retryCount": "0",
        "userAgent": "python-requests/2.28.1",
        "workerCap": "1.0",
        "status": "completed",
        "success": True,
        "running": False,
        "highPriority": False,
        "parentId": "611ef260-84ac-49c9-8de5-8e48062a7763",
        "nodeSize": 0,
        "metadataProcessed": False,
        "estFlexUnit": 0.027972380392156868,
        "objectRefId": "e07abd8b-cc30-4fd8-9159-c7654fe32ca8",
        "refId": "a52aa1f9-47f6-4041-9f6d-350389efe315",
        "combinedStatus": "completed",
        "elapsedTimeInSeconds": 44,
        "computeCost": 0.027972380392156868,
        "priorityScore": 0,
        "name": "case-fork",
        "id": "e07abd8b-cc30-4fd8-9159-c7654fe32ca8",
    }
    model = CaseMeta.parse_obj(resp)
    assert model.status.is_final()

    resp = {
        "caseId": "e07abd8b-cc30-4fd8-9159-c7654fe32ca8",
        "caseMeshId": "a52aa1f9-47f6-4041-9f6d-350389efe315",
        "caseName": "case-fork",
        "casePriority": None,
        "caseStatus": "completed",
        "caseStartTime": "2023-03-07T14:44:41.375Z",
        "caseSubmitTime": "2023-03-07T14:43:22.036Z",
        "caseFinishTime": "2023-03-07T14:45:25.828Z",
        "caseParentId": "611ef260-84ac-49c9-8de5-8e48062a7763",
        "userId": "AIDAXWCOWJGJINZTEEEIP",
        "caseTags": ["None"],
        "meshSize": 13560436,
        "nodesInfo": None,
        "solverVersion": "release-23.1.1.0",
        "worker": "B.XS.node5",
        "userEmail": "maciej@flexcompute.com",
        "estimationDuration": 0.0,
        "meshNodeSize": 113945,
        "realFlexUnit": 0.027972380392156868,
        "workerGroup": None,
        "estWorkUnit": 0.027972380392156868,
        "confidential": False,
        "storageSize": 67511271,
        "storageStatus": "STANDARD",
        "standardAge": 2,
        "storageClass": "STANDARD",
        "restoreAt": None,
        "deleted": False,
        "errorType": None,
        "numProcessors": "6",
        "retryCount": "0",
        "userAgent": "python-requests/2.28.1",
        "workerCap": "1.0",
        "status": "completed",
        "success": True,
        "running": False,
        "highPriority": False,
        "parentId": "611ef260-84ac-49c9-8de5-8e48062a7763",
        "nodeSize": 0,
        "metadataProcessed": False,
        "estFlexUnit": 0.027972380392156868,
        "objectRefId": "e07abd8b-cc30-4fd8-9159-c7654fe32ca8",
        "refId": "a52aa1f9-47f6-4041-9f6d-350389efe315",
        "combinedStatus": "completed",
        "elapsedTimeInSeconds": 44,
        "computeCost": 0.027972380392156868,
        "priorityScore": 0,
        "name": "case-fork",
        "id": "e07abd8b-cc30-4fd8-9159-c7654fe32ca8",
    }
    model = CaseMeta.parse_obj(resp)
    assert model.status.is_final()
