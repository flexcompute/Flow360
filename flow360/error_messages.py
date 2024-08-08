# pylint: disable=missing-function-docstring,invalid-name
"""
error messages
"""

not_a_cloud_resource = """\
Reference resource is not a cloud resource.
If a case was retried or forked from other case, submit the other case first before submitting this case.
"""


def change_solver_version_error(from_version, to_version):
    return f"""\
Cannot change solver version from parent to child.
Parent: solver_version={from_version}
Requested: solver_version={to_version}
You need to run sequence of all cases starting from mesh
"""


def shared_submit_warning(email):
    return f"""\
You are submitting a resource to a shared account {email}.
This message will not be shown again for this session if you confirm.
"""


def params_fetching_error(err_msg):
    return f"""\
There was a problem when fetching params for this case
{err_msg}
run .params_as_dict to get parameters as a raw dictionary
    """


def submit_reminder(class_name):
    return f"""\
Remember to submit your {class_name} to cloud to have it processed.
Please run .submit() after .create()
To suppress this message run: flow360 configure --suppress-submit-warning"""


def submit_warning(class_name):
    return f"\
WARNING: You have not submitted your {class_name} to cloud. \
It will not be processed. Please run .submit() after .create()"


do_not_modify_file_manually_msg = """\
The file was manually edited. It is recommended to not edit config files manually but to use:
>>> params = Flow360Params(filename)
>>> # edit your params here
>>> params.to_file(filename)
"""


use_unit_system_msg = """\
Params must be created with unit system context. For example:
>>> with SI_unit_system:
>>>     params = Flow360Params(
            ...
        )
"""

use_unit_system_for_simulation_msg = """\
SimulationParams must be created with a unit system context. For example:
>>> with SI_unit_system:
>>>     params = SimulationParams(
            ...
        )
"""


def unit_system_inconsistent_msg(kwarg_unit_system, context_unit_system):
    return f"""\
Tried to construct model with {kwarg_unit_system} inside {context_unit_system} context.
It can be caused by .copy() operation inside unit system context, \
or by providing unit system directly to the constructor.
"""
