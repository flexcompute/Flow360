# High level goal (general direction but not all in scope):
The SimulationParams schema should be moved into a dedicated Python package (I already created a temp folder for that)
we want to maintain the schema part separately from other parts (like the webAPI part of the Flow360 Python client)

At the same time the JSON schema dumped by Pydantic should be compatible with a "flavor" of JSON schema defined at 
`/disk2/ben/flex/frontend/workspace/packages/common-schema/`. THe main goal of this is to ensure Automatic frontend form
generation if we build our business model (like SimulationParams) schema on top of certain predefined building blocks
that the frontend already recognize.

To do this we plan (at least for now) to replace/modify all the shared building blocks such that they:
1. Encapsulate `unyt` and other non-Pydantic dependencies. Basically because the common-schema part does not want to
have very complex external dependencies. And also the schema generation itself should also not depend on package like
numpy or unit, because like you cannot directly pickle or serialize Numpy anyway, because the JSON format only has
support for like things like a list, float, integer, et cetera.

2. Then each of these like basic building block (referred to as primitives) would serve as a interface to the
common-schema. So basically the plan is to just completely overload their JSON schema generation such that at this
primitive building block level, the schema they generate is compatible with the common schema convention and the
requirements. Then we can ensure that whatever we build on top of these primitives can still satisfy the common schema
convention, so the frontend can automatically recognize what we are trying to build.

# High level implementation plan
So as I mentioned earlier, I already have this folder(`flow360/flow360_schema/`) created which serves as a storage for our separated schema package.
Eventually I want at the end of the entire grand project, We can directly publish this schema package or to import
that here `/disk2/ben/flex/share/flow360-schemas`

# In the scope of this particular plan that I want you to generate.
As first step I think we should migrate the unit primitives to this new package. Mostly here /disk2/ben/Flow360/flow360/component/simulation/unit_system.py.

I think I think as a start we should have some of the common schema primitive schemas being mapped into our unit primitive counterparts. For this, give me a list that you think are the most critical ones. For example the Scalar and the PositiveVector3 etc. The naming of these new types should be as close as to the counterparts in the common-schema environment.

This new primitive should be living in this new folder let's not try to change the existing primitive things that might just break the primitive and does the Python client let's just do everything in a new place without changing the existing one.

There are a few changes required when we do this:
1. I want to change the current implementation where the validation function is centralized (AKA there is only one) and capable of handling all kinds of validation criterion namely for example the length of the array as well as the positivity as well as the magnitude etc. The more cleaner way in my opinion is to have each of derive the class (For example Positive) defining their own validate() function but each criteria is implemented to a particular standalone function and each new primitive class essentially just call a composition of these single criterion validation functions instead of trying to squeeze all validation into one function.
2. Right now the Jason data or the serialization would dump both the "value" part as well as the "unit" part into the Jason. However if you take a look at the common schema you will figure out that the new convention is that way we only store the value. The unit instead is a "compile-time" metadata. For now we use SI units. So then this basically dictates that whatever unyt object we have in the memory we need to convert to the designated unit before serializing it otherwise the data would be incomplete.
3. On the other hand when we do deserialization, it would essentially breaks into two parts so the 1st part is just loading number which Pydantic probably can do it natively the other part is the validation and as I mentioned earlier this validated function even though it depends on Unyt because it has to like convert a number into a Unyt object but this conversion or the dependency on the Unyt package has to be encapsulated so that the JSON schema generation does not require Unyt package to be installed. You can technically try just lazy import but I'm not sure if there's any other like a more way of doing this but I think lazy import is fine.
4. Another side request was that previously uh zero is not considered a valid input for dimension values so in these new primitive I was hoping that zero without unit is accepted for most of the units input Uh perhaps other than the temperature because in temperature zero might mean different things depending on which unit you are trying to use but for every other dimension or unit I think zero has universal meaning so we can't allow zero.

Write a plan and break into steps. Try not to make any assumption and always ask me if you are uncertain.