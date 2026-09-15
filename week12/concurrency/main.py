import asyncio
import time


async def task(name: str, duration: float) -> str:
    print(f"[START] {name}")

    await asyncio.sleep(duration)

    print(f"[DONE]  {name}")

    return name


async def sequential():
    start = time.perf_counter()

    await task("A", 2)
    await task("B", 2)
    await task("C", 2)

    elapsed = time.perf_counter() - start

    print(f"\nSequential time: {elapsed:.2f}s")


async def concurrent():
    start = time.perf_counter()

    await asyncio.gather(
        task("A", 2),
        task("B", 2),
        task("C", 2),
    )

    elapsed = time.perf_counter() - start

    print(f"\nConcurrent time: {elapsed:.2f}s")


async def main():
    print("================================")
    print("SEQUENTIAL")
    print("================================")

    await sequential()

    print("\n================================")
    print("CONCURRENT")
    print("================================")

    await concurrent()


if __name__ == "__main__":
    asyncio.run(main())
