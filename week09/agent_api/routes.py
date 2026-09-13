from fastapi import APIRouter, HTTPException, Request

from .schemas import AgentRequest, AgentResponse
from .service import get_trace, run_agent


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
    }


@router.post("/agent/run", response_model=AgentResponse)
def run_agent_endpoint(
    request: Request,
    agent_request: AgentRequest,
) -> AgentResponse:

    request_id = request.state.request_id

    try:
        result = run_agent(
            goal=agent_request.goal,
            request_id=request_id,
        )

        return AgentResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Agent execution failed.",
        )


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