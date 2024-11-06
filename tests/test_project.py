from flow360.component.project import Project, ProjectMeta


def test_project(mock_id, mock_response):
    Project(
        metadata=ProjectMeta(
            user_id="user_id",
        )
    )
