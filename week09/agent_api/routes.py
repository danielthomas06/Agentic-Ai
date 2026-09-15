import traceback

from fastapi import APIRouter, Depends, HTTPException, Request

from .auth import authenticate
from .schemas import AgentRequest, AgentResponse
from .service import get_trace, run_agent


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
    }


@router.post(
    "/agent/run",
    response_model=AgentResponse,
)
def run_agent_endpoint(
    request: Request,
    agent_request: AgentRequest,
    authenticated_role: str = Depends(authenticate),
) -> AgentResponse:

    request_id = request.state.request_id

    try:
        result = run_agent(
            goal=agent_request.goal,
            request_id=request_id,
            agent_role=authenticated_role,
        )

        return AgentResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        print(
            f"[AGENT ERROR] "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="Agent execution failed.",
        ) from exc


@router.get("/agent/trace/{request_id}")
def get_agent_trace(request_id: str) -> dict:
    trace = get_trace(request_id)

    if trace is None:
        raise HTTPException(
            status_code=404,
            detail="Trace not found.",
        )

    return {
        "request_id": request_id,
        "trace": trace,
    }