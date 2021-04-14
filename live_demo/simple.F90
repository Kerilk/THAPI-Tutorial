PROGRAM target_teams_distribute_parallel_do
  implicit none
  INTEGER :: N0 = 32768
  INTEGER :: i0
  REAL, ALLOCATABLE :: src(:)
  REAL, ALLOCATABLE :: dst(:)
  ALLOCATE(dst(N0), src(N0) )
  CALL RANDOM_NUMBER(src)
  !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO map(to: src) map(from: dst)
  DO i0 = 1, N0
    dst(i0) = src(i0)
  END DO
END PROGRAM target_teams_distribute_parallel_do
