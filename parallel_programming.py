import multiprocessing as mp
import random
import string



def sieve_prime(n, output, job):
    prime = []
    sieve = [True] * (n + 1)
    for p in range(2, n + 1):
        if sieve[p]:
            prime.append(p)
            for i in range(p * p, n + 1, p):
                sieve[i] = False

    return output.put((job, str(len(prime))))

output = mp.Queue()
processes = [mp.Process(target=sieve_prime, args=(random.randint(10000,100000), output, 4)),
             mp.Process(target=sieve_prime, args=(random.randint(5000,10000), output, 3)),
             mp.Process(target=sieve_prime, args=(random.randint(1000, 5000), output, 2)),
             mp.Process(target=sieve_prime, args=(random.randint(100000, 1000000), output, 1)),
             mp.Process(target=sieve_prime, args=(random.randint(100000, 5000000), output, 0))]

for p in processes:
    p.start()

for p in processes:
    p.join()

results = [output.get() for p in processes]
for result in results:
    print("process id {} - results = {}".format(str(result[0]), result[1]))
