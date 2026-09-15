import asyncio
import time


CONCURRENCY_LIMIT = 3


async def process_request(
    request_id: int,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        start = time.perf_counter()

        print(f"[START] Request {request_id}")

        # Simulate expensive agent/LLM work.
        await asyncio.sleep(2)

        elapsed = time.perf_counter() - start

        print(
            f"[DONE]  Request {request_id} "
            f"({elapsed:.2f}s)"
        )


async def main():
    total_requests = 10

    semaphore = asyncio.Semaphore(
        CONCURRENCY_LIMIT
    )

    start = time.perf_counter()

    await asyncio.gather(
        *[
            process_request(
                request_id=i,
                semaphore=semaphore,
            )
            for i in range(1, total_requests + 1)
        ]
    )

    elapsed = time.perf_counter() - start

    print("\n==============================")
    print("RESULT")
    print("==============================")
    print(f"Requests: {total_requests}")
    print(f"Concurrency limit: {CONCURRENCY_LIMIT}")
    print(f"Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
